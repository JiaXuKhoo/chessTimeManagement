"""
run_tournament.py
=================
Compares the ML-driven TokenBucket controller against three baseline
allocation policies under equal total node budgets.

Policies tested:
  1. FixedUniform   — total_budget / 50 every move (no adaptation)
  2. SolakVuckovic  — remaining_budget / estimated_moves_left
                      where moves_left uses the Šolak & Vučković (2009) formula
  3. Hyatt          — front-loaded variant of (2) using Hyatt (1984) factor
  4. TokenBucket    — ML classifier + token bucket (our system)

Supports both:
  - static-only models
  - probed-feature models matching getprobedfeatures.py

Each matchup: 100 openings × 2 colours = 200 games.
Each matchup is run in both discrete (snapped to buckets) and continuous mode.

Usage:
  python run_tournament.py
"""

import joblib
import chess
import chess.engine
import pandas as pd
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

# =========================================================
# Config
# =========================================================

STOCKFISH_PATH = r"stockfish"

#   gbt_static_tol20.joblib OR
#   gbt_probe_tol20.joblib
MODEL_PATH = "gbt_probe_tol20.joblib"
OPENINGS_FILE = "openings_100.txt"
# tournament_results_static OR
# tournament_results_probe
RESULTS_DIR = "tournament_results_probe"

TOTAL_NODE_BUDGET = 10_000_000
BUCKETS = [25_000, 100_000, 400_000, 1_600_000]

# Probe config (used only if model requires probe_* features)
PROBE_NODES = 5000

# =========================================================
# Adjudication rules
# =========================================================
#
# DRAW RULE — follows TCEC-like logic:
#   From move 40 onward, if the eval stays within ±5cp for 10
#   consecutive plies (5 full moves), adjudicate as a draw.
#   A capture or pawn advance resets the counter.
#
# RESIGN RULE — justified for same-engine experimental play:
#   If either side's eval reaches ≥500cp (in its favour) for 4
#   consecutive plies, the losing side is adjudicated as resigning.
#
# BUDGET EXHAUSTION — time forfeit:
#   If a side's node budget reaches zero before the game ends
#   naturally, that side loses.
MAX_PLIES = 400
DRAW_MOVE_THRESHOLD = 40
DRAW_CP_THRESHOLD = 5
DRAW_CONSECUTIVE = 10
RESIGN_CP_THRESHOLD = 500
RESIGN_CONSECUTIVE = 4

HASH_MB = 128
THREADS = 1

# =========================================================
# Piece values
# =========================================================

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
}


# =========================================================
# Feature extraction
# =========================================================

def extract_static_features(board: chess.Board) -> Dict[str, int]:
    feats = {}
    feats["num_legal_moves"] = board.legal_moves.count()
    feats["is_check"] = int(board.is_check())

    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        name = PIECE_NAMES[pt]
        feats[f"white_{name}_count"] = len(board.pieces(pt, chess.WHITE))
        feats[f"black_{name}_count"] = len(board.pieces(pt, chess.BLACK))

    total_mat, mat_imbal = 0, 0
    for pt, v in PIECE_VALUES.items():
        w = len(board.pieces(pt, chess.WHITE))
        b = len(board.pieces(pt, chess.BLACK))
        total_mat += v * (w + b)
        mat_imbal += v * (w - b)

    feats["total_material"] = total_mat
    feats["material_imbalance"] = mat_imbal
    feats["total_pieces"] = sum(
        1 for p in board.piece_map().values() if p.piece_type != chess.KING
    )
    feats["side_to_move_white"] = int(board.turn == chess.WHITE)
    feats["white_can_castle_kingside"] = int(board.has_kingside_castling_rights(chess.WHITE))
    feats["white_can_castle_queenside"] = int(board.has_queenside_castling_rights(chess.WHITE))
    feats["black_can_castle_kingside"] = int(board.has_kingside_castling_rights(chess.BLACK))
    feats["black_can_castle_queenside"] = int(board.has_queenside_castling_rights(chess.BLACK))
    feats["halfmove_clock"] = board.halfmove_clock

    nc, nch, np = 0, 0, 0
    for move in board.legal_moves:
        if board.is_capture(move):
            nc += 1
        if move.promotion:
            np += 1
        if board.gives_check(move):
            nch += 1
    feats["num_captures"] = nc
    feats["num_checks"] = nch
    feats["num_promotions"] = np

    for pt, fname in [
        (chess.KNIGHT, "knight_mobility"),
        (chess.BISHOP, "bishop_mobility"),
        (chess.ROOK, "rook_mobility"),
        (chess.QUEEN, "queen_mobility"),
    ]:
        t = 0
        occ = board.occupied_co[board.turn]
        for sq in board.pieces(pt, board.turn):
            t += len(board.attacks(sq) & ~occ)
        feats[fname] = t

    enemy_king = board.king(not board.turn)
    feats["num_attackers_on_king"] = (
        len(board.attackers(board.turn, enemy_king)) if enemy_king else 0
    )

    pin_count = 0
    for sq, piece in board.piece_map().items():
        if piece.color == board.turn and piece.piece_type != chess.KING:
            if board.is_pinned(board.turn, sq):
                pin_count += 1
    feats["num_pinned_pieces"] = pin_count

    return feats


# =========================================================
# Probe feature extraction
# =========================================================

def extract_score_info(
    score_obj: Optional[chess.engine.PovScore]
) -> Tuple[Optional[str], Optional[int]]:
    if score_obj is None:
        return None, None

    white_score = score_obj.white()

    if white_score.is_mate():
        mate_val = white_score.mate()
        return "mate", mate_val if mate_val is not None else None

    cp = white_score.score()
    return "cp", cp if cp is not None else None


def score_to_cp_for_gap(
    score_obj: Optional[chess.engine.PovScore]
) -> Optional[int]:
    if score_obj is None:
        return None

    white_score = score_obj.white()
    if white_score.is_mate():
        return None

    return white_score.score()


def extract_probe_features(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    probe_nodes: int = PROBE_NODES,
) -> Dict:
    feats: Dict = {}

    infos = engine.analyse(
        board,
        chess.engine.Limit(nodes=probe_nodes),
        multipv=2
    )

    if isinstance(infos, list):
        pv1 = infos[0] if len(infos) >= 1 else {}
        pv2 = infos[1] if len(infos) >= 2 else {}
    else:
        pv1 = infos
        pv2 = {}

    score_type, score_val = extract_score_info(pv1.get("score"))
    feats["probe_score_type"] = score_type
    feats["probe_score_cp"] = score_val if score_type == "cp" else None
    feats["probe_abs_score_cp"] = abs(score_val) if score_type == "cp" and score_val is not None else None
    feats["probe_is_mate"] = int(score_type == "mate")
    feats["probe_mate_value"] = score_val if score_type == "mate" else None

    feats["probe_depth"] = pv1.get("depth")
    feats["probe_seldepth"] = pv1.get("seldepth")

    cp1 = score_to_cp_for_gap(pv1.get("score"))
    cp2 = score_to_cp_for_gap(pv2.get("score")) if pv2 else None
    feats["probe_best_second_gap"] = (cp1 - cp2) if (cp1 is not None and cp2 is not None) else 0

    feats["probe_nodes_requested"] = probe_nodes
    feats["probe_nodes_reported"] = pv1.get("nodes")

    return feats


# =========================================================
# Moves-left estimator: Šolak & Vučković (2009)
# =========================================================

def get_total_material(board: chess.Board) -> int:
    """Sum of piece values for both sides, excluding kings."""
    x = 0
    for pt, v in PIECE_VALUES.items():
        x += v * len(board.pieces(pt, chess.WHITE))
        x += v * len(board.pieces(pt, chess.BLACK))
    return x


def estimate_moves_left(board: chess.Board) -> int:
    """
    Šolak & Vučković (2009) formula.
    Maps total material x to remaining half-moves y, returns y/2 (per-side).

      y = x + 10              if x < 20
      y = (3/8)x + 22         if 20 <= x <= 60
      y = (5/4)x - 30         if x > 60
    """
    x = get_total_material(board)

    if x < 20:
        y = x + 10
    elif x <= 60:
        y = (3 / 8) * x + 22
    else:
        y = (5 / 4) * x - 30

    return max(1, int(y / 2))


# =========================================================
# Snapping to discrete buckets
# =========================================================

def snap_to_bucket(nodes: float) -> int:
    """Largest discrete bucket that fits within the given node count."""
    affordable = [b for b in BUCKETS if b <= nodes]
    if affordable:
        return max(affordable)
    return min(int(nodes), BUCKETS[0])


# =========================================================
# Allocation policies
# =========================================================

class AllocationPolicy(ABC):
    """Base class for all allocation policies."""

    def __init__(self, name: str, total_budget: int, use_discrete: bool):
        self.name = name
        self.total_budget = total_budget
        self.remaining_budget = total_budget
        self.use_discrete = use_discrete

    @abstractmethod
    def decide_nodes(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        ply: int
    ) -> int:
        pass

    def consume(self, nodes_used: int):
        self.remaining_budget = max(0, self.remaining_budget - nodes_used)

    def finalise(self, raw: float) -> int:
        """Clamp to remaining budget and optionally snap to a discrete bucket."""
        raw = min(raw, self.remaining_budget)
        if self.use_discrete:
            raw = max(BUCKETS[0], raw)
            return snap_to_bucket(raw)
        return max(1, int(raw))

    def reset(self):
        self.remaining_budget = self.total_budget


class FixedUniformPolicy(AllocationPolicy):
    """
    Simplest possible baseline: total_budget / 50 every single move.
    No adaptation to board state or remaining budget whatsoever.
    """

    def __init__(self, total_budget: int, use_discrete: bool):
        super().__init__("FixedUniform", total_budget, use_discrete)
        self.fixed_allocation = total_budget / 50

    def decide_nodes(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        ply: int
    ) -> int:
        raw = min(self.fixed_allocation, self.remaining_budget)
        return self.finalise(raw)


class SolakVuckovicPolicy(AllocationPolicy):
    """
    Divides remaining budget evenly among estimated remaining moves.
    """

    def __init__(self, total_budget: int, use_discrete: bool):
        super().__init__("SolakVuckovic", total_budget, use_discrete)

    def decide_nodes(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        ply: int
    ) -> int:
        moves_left = estimate_moves_left(board)
        raw = self.remaining_budget / moves_left
        return self.finalise(raw)


class HyattPolicy(AllocationPolicy):
    """
    Hyatt (1984) 'Using Time Wisely' style front-loaded scheme.
    """

    def __init__(self, total_budget: int, use_discrete: bool):
        super().__init__("Hyatt", total_budget, use_discrete)

    def decide_nodes(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        ply: int
    ) -> int:
        moves_left = estimate_moves_left(board)
        target = self.remaining_budget / moves_left

        side_moves_played = ply // 2
        n = min(side_moves_played, 10)
        factor = 2.0 - n / 10.0

        raw = factor * target
        return self.finalise(raw)


class TokenBucketPolicy(AllocationPolicy):
    """
    Supports both static-only and probed-feature models.

    If any feature column begins with 'probe_', a small exploratory probe
    search is run before prediction.
    """

    def __init__(
        self,
        model_path: str,
        total_budget: int,
        use_discrete: bool,
        burst_cap: int = BUCKETS[-1] * 2
    ):
        super().__init__("TokenBucket", total_budget, use_discrete)

        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_cols = payload["feature_cols"]

        self.uses_probe = any(col.startswith("probe_") for col in self.feature_cols)

        self.burst_cap = burst_cap
        self.tokens = float(burst_cap)
        self.refill_rate = total_budget / 50

    def reset(self):
        super().reset()
        self.tokens = float(self.burst_cap)
        self.refill_rate = self.total_budget / 50

    def build_feature_row(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board
    ) -> pd.DataFrame:
        feats = extract_static_features(board)

        if self.uses_probe:
            probe_feats = extract_probe_features(engine, board, probe_nodes=PROBE_NODES)
            feats.update(probe_feats)

        row = {}
        for col in self.feature_cols:
            val = feats.get(col, 0)

            # Defensive numeric sanitization
            if val is None:
                val = 0
            elif isinstance(val, str):
                val = 0

            row[col] = val

        return pd.DataFrame([row], columns=self.feature_cols)

    def decide_nodes(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        ply: int
    ) -> int:
        moves_left = estimate_moves_left(board)
        self.refill_rate = self.remaining_budget / moves_left

        X = self.build_feature_row(engine, board)
        predicted = int(self.model.predict(X)[0])

        spendable = min(predicted, self.tokens, self.remaining_budget)
        return self.finalise(spendable)

    def consume(self, nodes_used: int):
        super().consume(nodes_used)
        self.tokens = max(0.0, self.tokens - nodes_used)
        self.tokens = min(self.tokens + self.refill_rate, float(self.burst_cap))


# =========================================================
# Game result
# =========================================================

@dataclass
class GameResult:
    opening_idx: int
    opening_fen: str
    white_policy: str
    black_policy: str
    discrete: bool
    result: str
    termination: str
    total_plies: int
    white_budget_remaining: int
    black_budget_remaining: int
    white_nodes_total: int
    black_nodes_total: int


# =========================================================
# Game runner
# =========================================================

def play_one_game(
    engine: chess.engine.SimpleEngine,
    opening_fen: str,
    white_policy: AllocationPolicy,
    black_policy: AllocationPolicy,
    opening_idx: int,
) -> GameResult:
    """
    Play one game. Each side uses its own allocation policy and budget.
    Both sides share the same Stockfish engine for move computation.
    """
    board = chess.Board(opening_fen)
    white_policy.reset()
    black_policy.reset()

    w_nodes = 0
    b_nodes = 0

    draw_counter = 0
    resign_w = 0
    resign_b = 0

    ply = 0
    termination = "max_plies"
    budget_forfeit_side = None

    while not board.is_game_over() and ply < MAX_PLIES:
        policy = white_policy if board.turn == chess.WHITE else black_policy

        if policy.remaining_budget <= 0:
            termination = "budget_forfeit"
            budget_forfeit_side = board.turn
            break

        nodes = policy.decide_nodes(engine, board, ply)
        nodes = max(1, min(nodes, policy.remaining_budget))

        info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
        pv = info.get("pv", [])
        if not pv:
            termination = "no_pv"
            break

        move = pv[0]
        nodes_used = int(info.get("nodes", nodes))

        if board.turn == chess.WHITE:
            w_nodes += nodes_used
        else:
            b_nodes += nodes_used

        policy.consume(nodes_used)

        is_capture = board.is_capture(move)
        piece = board.piece_at(move.from_square)
        is_pawn_move = piece is not None and piece.piece_type == chess.PAWN

        score = info.get("score")
        if score is not None:
            cp = score.white().score()
            if cp is not None:
                full_move = ply // 2 + 1

                # Draw rule
                if is_capture or is_pawn_move:
                    draw_counter = 0
                elif full_move >= DRAW_MOVE_THRESHOLD and abs(cp) <= DRAW_CP_THRESHOLD:
                    draw_counter += 1
                else:
                    draw_counter = 0

                if draw_counter >= DRAW_CONSECUTIVE:
                    board.push(move)
                    ply += 1
                    termination = "draw_adjudication"
                    break

                # Resign rule
                if cp <= -RESIGN_CP_THRESHOLD:
                    resign_w += 1
                else:
                    resign_w = 0

                if cp >= RESIGN_CP_THRESHOLD:
                    resign_b += 1
                else:
                    resign_b = 0

                if resign_w >= RESIGN_CONSECUTIVE:
                    board.push(move)
                    ply += 1
                    termination = "resign_adjudication"
                    break

                if resign_b >= RESIGN_CONSECUTIVE:
                    board.push(move)
                    ply += 1
                    termination = "resign_adjudication"
                    break

        board.push(move)
        ply += 1

    if board.is_game_over():
        result = board.result()
        if board.is_checkmate():
            termination = "checkmate"
        elif board.is_stalemate():
            termination = "stalemate"
        else:
            termination = "draw_rule"
    elif termination == "draw_adjudication":
        result = "1/2-1/2"
    elif termination == "resign_adjudication":
        result = "0-1" if resign_w >= RESIGN_CONSECUTIVE else "1-0"
    elif termination == "budget_forfeit":
        result = "0-1" if budget_forfeit_side == chess.WHITE else "1-0"
    else:
        result = "*"

    return GameResult(
        opening_idx=opening_idx,
        opening_fen=opening_fen,
        white_policy=white_policy.name,
        black_policy=black_policy.name,
        discrete=white_policy.use_discrete,
        result=result,
        termination=termination,
        total_plies=ply,
        white_budget_remaining=white_policy.remaining_budget,
        black_budget_remaining=black_policy.remaining_budget,
        white_nodes_total=w_nodes,
        black_nodes_total=b_nodes,
    )


# =========================================================
# Matchup runner
# =========================================================

def run_matchup(
    engine: chess.engine.SimpleEngine,
    openings: List[str],
    make_a,
    make_b,
    matchup_name: str,
    use_discrete: bool,
) -> List[GameResult]:
    """
    Run one matchup: each opening played twice with colour reversal.
    A is the protagonist (controller), B is the baseline.
    """
    mode = "discrete" if use_discrete else "continuous"
    total = len(openings) * 2

    print(f"\n{'='*60}")
    print(f"{matchup_name} ({mode}) — {total} games")
    print(f"{'='*60}")

    results = []

    for idx, fen in enumerate(openings):
        # Game 1: A=White, B=Black
        a = make_a(use_discrete)
        b = make_b(use_discrete)
        g1 = play_one_game(engine, fen, a, b, idx)
        results.append(g1)

        # Game 2: B=White, A=Black
        a = make_a(use_discrete)
        b = make_b(use_discrete)
        g2 = play_one_game(engine, fen, b, a, idx)
        results.append(g2)

        if (idx + 1) % 10 == 0:
            a_name = make_a(use_discrete).name
            a_w = sum(
                1 for r in results
                if (r.white_policy == a_name and r.result == "1-0")
                or (r.black_policy == a_name and r.result == "0-1")
            )
            a_l = sum(
                1 for r in results
                if (r.white_policy == a_name and r.result == "0-1")
                or (r.black_policy == a_name and r.result == "1-0")
            )
            d = sum(1 for r in results if r.result == "1/2-1/2")
            print(f"  [{idx+1}/{len(openings)}] {a_name}: +{a_w} ={d} -{a_l}")

    return results


def summarise(results: List[GameResult], a_name: str):
    """Print summary from policy A's perspective."""
    a_w = sum(
        1 for r in results
        if (r.white_policy == a_name and r.result == "1-0")
        or (r.black_policy == a_name and r.result == "0-1")
    )
    a_l = sum(
        1 for r in results
        if (r.white_policy == a_name and r.result == "0-1")
        or (r.black_policy == a_name and r.result == "1-0")
    )
    d = sum(1 for r in results if r.result == "1/2-1/2")
    u = sum(1 for r in results if r.result == "*")
    total = len(results)
    score = a_w + 0.5 * d
    pct = score / total * 100 if total else 0

    print(f"\n  {a_name}: +{a_w} ={d} -{a_l} (*{u})  "
          f"Score: {score}/{total} ({pct:.1f}%)")

    terms = {}
    for r in results:
        terms[r.termination] = terms.get(r.termination, 0) + 1
    print(f"  Terminations: {terms}")


def save_csv(results: List[GameResult], path: str):
    """Write detailed results to CSV."""
    fields = [
        "opening_idx", "opening_fen", "white_policy", "black_policy",
        "discrete", "result", "termination", "total_plies",
        "white_budget_remaining", "black_budget_remaining",
        "white_nodes_total", "black_nodes_total",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "opening_idx": r.opening_idx,
                "opening_fen": r.opening_fen,
                "white_policy": r.white_policy,
                "black_policy": r.black_policy,
                "discrete": r.discrete,
                "result": r.result,
                "termination": r.termination,
                "total_plies": r.total_plies,
                "white_budget_remaining": r.white_budget_remaining,
                "black_budget_remaining": r.black_budget_remaining,
                "white_nodes_total": r.white_nodes_total,
                "black_nodes_total": r.black_nodes_total,
            })
    print(f"  Saved: {path}")


# =========================================================
# Main
# =========================================================

def main():
    if not os.path.exists(OPENINGS_FILE):
        print(f"ERROR: {OPENINGS_FILE} not found. Run generate_openings.py first.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found.")
        return

    with open(OPENINGS_FILE, "r", encoding="utf-8") as f:
        openings = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(openings)} openings")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    def make_controller(disc):
        return TokenBucketPolicy(MODEL_PATH, TOTAL_NODE_BUDGET, disc)

    def make_fixed(disc):
        return FixedUniformPolicy(TOTAL_NODE_BUDGET, disc)

    def make_solak(disc):
        return SolakVuckovicPolicy(TOTAL_NODE_BUDGET, disc)

    def make_hyatt(disc):
        return HyattPolicy(TOTAL_NODE_BUDGET, disc)

    matchups = [
        ("TokenBucket_vs_FixedUniform",   make_controller, make_fixed),
        ("TokenBucket_vs_SolakVuckovic",  make_controller, make_solak),
        ("TokenBucket_vs_Hyatt",          make_controller, make_hyatt),
    ]

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": THREADS, "Hash": HASH_MB})

    t0 = time.time()

    try:
        for name, make_a, make_b in matchups:
            for use_discrete in [True, False]:
                mode = "discrete" if use_discrete else "continuous"

                results = run_matchup(
                    engine, openings, make_a, make_b, name, use_discrete,
                )

                a_name = make_a(use_discrete).name
                summarise(results, a_name)

                csv_path = os.path.join(RESULTS_DIR, f"{name}_{mode}.csv")
                save_csv(results, csv_path)
    finally:
        engine.quit()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Tournament complete in {elapsed/60:.1f} minutes")
    print(f"Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()