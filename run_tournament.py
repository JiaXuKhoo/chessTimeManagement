"""
run_tournament.py
==========================
Parallel tournament runner for comparing the ML-driven TokenBucket controller
against three baseline allocation policies under equal total node budgets.

Policies tested:
  1. FixedUniform   — total_budget / 50 every move (no adaptation)
  2. SolakVuckovic  — remaining_budget / estimated_moves_left
  3. Hyatt          — front-loaded variant of Solak-Vuckovic
  4. TokenBucket    — ML classifier + token bucket

Design:
- Each worker owns its own Stockfish engine instance.
- Each job = one opening + one matchup + one mode (discrete/continuous).
- Each job runs 2 games (both colour assignments).
- Results are aggregated and saved per matchup/mode CSV.
"""

import csv
import os
import time
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

import chess
import chess.engine
import joblib
import pandas as pd


# =========================================================
# Config
# =========================================================

STOCKFISH_PATH = r"stockfish"
MODEL_PATH = "gbt_static_tol20.joblib"
OPENINGS_FILE = "openings_100.txt"
RESULTS_DIR = "tournament_results"

TOTAL_NODE_BUDGET = 10_000_000
BUCKETS = [25_000, 100_000, 400_000, 1_600_000]

# Adjudication thresholds
MAX_PLIES = 300
DRAW_MOVE_THRESHOLD = 40
DRAW_CP_THRESHOLD = 8
DRAW_CONSECUTIVE = 8
RESIGN_CP_THRESHOLD = 500
RESIGN_CONSECUTIVE = 4

# Per-engine settings
HASH_MB = 128
THREADS = 1

# Parallelism
NUM_WORKERS = max(1, (os.cpu_count() or 2) - 1)


# =========================================================
# Piece values and names
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

    total_mat = 0
    mat_imbal = 0
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

    nc = 0
    nch = 0
    npromo = 0
    for move in board.legal_moves:
        if board.is_capture(move):
            nc += 1
        if move.promotion:
            npromo += 1
        if board.gives_check(move):
            nch += 1

    feats["num_captures"] = nc
    feats["num_checks"] = nch
    feats["num_promotions"] = npromo

    for pt, fname in [
        (chess.KNIGHT, "knight_mobility"),
        (chess.BISHOP, "bishop_mobility"),
        (chess.ROOK, "rook_mobility"),
        (chess.QUEEN, "queen_mobility"),
    ]:
        t = 0
        own_occ = board.occupied_co[board.turn]
        for sq in board.pieces(pt, board.turn):
            t += len(board.attacks(sq) & ~own_occ)
        feats[fname] = t

    enemy_king = board.king(not board.turn)
    feats["num_attackers_on_king"] = (
        len(board.attackers(board.turn, enemy_king)) if enemy_king is not None else 0
    )

    pin_count = 0
    for sq, piece in board.piece_map().items():
        if piece.color == board.turn and piece.piece_type != chess.KING:
            if board.is_pinned(board.turn, sq):
                pin_count += 1
    feats["num_pinned_pieces"] = pin_count

    return feats


# =========================================================
# Moves-left estimator
# =========================================================

def get_total_material(board: chess.Board) -> int:
    x = 0
    for pt, v in PIECE_VALUES.items():
        x += v * len(board.pieces(pt, chess.WHITE))
        x += v * len(board.pieces(pt, chess.BLACK))
    return x


def estimate_moves_left(board: chess.Board) -> int:
    """
    Šolak & Vučković-style material-based moves-left estimate.
    Returns per-side remaining moves.
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
# Helpers
# =========================================================

def snap_to_bucket(nodes: float) -> int:
    affordable = [b for b in BUCKETS if b <= nodes]
    if affordable:
        return max(affordable)
    return min(int(nodes), BUCKETS[0])


def open_engine() -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": THREADS, "Hash": HASH_MB})
    return engine


def close_engine(engine: Optional[chess.engine.SimpleEngine]) -> None:
    if engine is not None:
        try:
            engine.quit()
        except Exception:
            pass


# =========================================================
# Allocation policies
# =========================================================

class AllocationPolicy(ABC):
    def __init__(self, name: str, total_budget: int, use_discrete: bool):
        self.name = name
        self.total_budget = total_budget
        self.remaining_budget = total_budget
        self.use_discrete = use_discrete

    @abstractmethod
    def decide_nodes(self, board: chess.Board, ply: int) -> int:
        raise NotImplementedError

    def consume(self, nodes_used: int):
        self.remaining_budget = max(0, self.remaining_budget - nodes_used)

    def finalise(self, raw: float) -> int:
        raw = max(BUCKETS[0], min(raw, self.remaining_budget))
        if self.use_discrete:
            return snap_to_bucket(raw)
        return max(1, int(raw))

    def reset(self):
        self.remaining_budget = self.total_budget


class FixedUniformPolicy(AllocationPolicy):
    def __init__(self, total_budget: int, use_discrete: bool):
        super().__init__("FixedUniform", total_budget, use_discrete)
        self.fixed_allocation = total_budget / 50

    def decide_nodes(self, board: chess.Board, ply: int) -> int:
        raw = min(self.fixed_allocation, self.remaining_budget)
        return self.finalise(raw)


class SolakVuckovicPolicy(AllocationPolicy):
    def __init__(self, total_budget: int, use_discrete: bool):
        super().__init__("SolakVuckovic", total_budget, use_discrete)

    def decide_nodes(self, board: chess.Board, ply: int) -> int:
        moves_left = estimate_moves_left(board)
        raw = self.remaining_budget / moves_left
        return self.finalise(raw)


class HyattPolicy(AllocationPolicy):
    def __init__(self, total_budget: int, use_discrete: bool):
        super().__init__("Hyatt", total_budget, use_discrete)

    def decide_nodes(self, board: chess.Board, ply: int) -> int:
        moves_left = estimate_moves_left(board)
        target = self.remaining_budget / moves_left

        side_moves_played = ply // 2
        n = min(side_moves_played, 10)
        factor = 2.0 - n / 10.0

        raw = factor * target
        return self.finalise(raw)


class TokenBucketPolicy(AllocationPolicy):
    def __init__(
        self,
        model_path: str,
        total_budget: int,
        use_discrete: bool,
        burst_cap: int = BUCKETS[-1] * 2,
    ):
        super().__init__("TokenBucket", total_budget, use_discrete)

        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_cols = payload["feature_cols"]

        self.burst_cap = burst_cap
        self.tokens = float(burst_cap)
        self.refill_rate = total_budget / 50

    def reset(self):
        super().reset()
        self.tokens = float(self.burst_cap)
        self.refill_rate = self.total_budget / 50

    def decide_nodes(self, board: chess.Board, ply: int) -> int:
        moves_left = estimate_moves_left(board)
        self.refill_rate = self.remaining_budget / moves_left

        feats = extract_static_features(board)
        row = {col: feats.get(col, 0) for col in self.feature_cols}
        X = pd.DataFrame([row], columns=self.feature_cols)
        predicted = int(self.model.predict(X)[0])

        spendable = min(predicted, self.tokens, self.remaining_budget)
        return self.finalise(spendable)

    def consume(self, nodes_used: int):
        super().consume(nodes_used)
        self.tokens = max(0.0, self.tokens - nodes_used)
        self.tokens = min(self.tokens + self.refill_rate, float(self.burst_cap))


# =========================================================
# Policy registry
# =========================================================

def make_policy(policy_name: str, use_discrete: bool) -> AllocationPolicy:
    if policy_name == "TokenBucket":
        return TokenBucketPolicy(MODEL_PATH, TOTAL_NODE_BUDGET, use_discrete)
    if policy_name == "FixedUniform":
        return FixedUniformPolicy(TOTAL_NODE_BUDGET, use_discrete)
    if policy_name == "SolakVuckovic":
        return SolakVuckovicPolicy(TOTAL_NODE_BUDGET, use_discrete)
    if policy_name == "Hyatt":
        return HyattPolicy(TOTAL_NODE_BUDGET, use_discrete)
    raise ValueError(f"Unknown policy: {policy_name}")


MATCHUP_REGISTRY = {
    "TokenBucket_vs_FixedUniform": ("TokenBucket", "FixedUniform"),
    "TokenBucket_vs_SolakVuckovic": ("TokenBucket", "SolakVuckovic"),
    "TokenBucket_vs_Hyatt": ("TokenBucket", "Hyatt"),
}


# =========================================================
# Result structures
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
    matchup_name: str


@dataclass
class MatchJob:
    matchup_name: str
    opening_idx: int
    opening_fen: str
    use_discrete: bool


# =========================================================
# Game runner
# =========================================================

def play_one_game(
    engine: chess.engine.SimpleEngine,
    opening_fen: str,
    white_policy: AllocationPolicy,
    black_policy: AllocationPolicy,
    opening_idx: int,
    matchup_name: str,
) -> GameResult:
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

    while not board.is_game_over() and ply < MAX_PLIES:
        policy = white_policy if board.turn == chess.WHITE else black_policy

        if policy.remaining_budget <= 0:
            termination = "budget_exhausted"
            break

        nodes = policy.decide_nodes(board, ply)
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

        score = info.get("score")
        if score is not None:
            cp = score.white().score()
            if cp is not None:
                full_move = ply // 2 + 1

                if full_move >= DRAW_MOVE_THRESHOLD and abs(cp) <= DRAW_CP_THRESHOLD:
                    draw_counter += 1
                else:
                    draw_counter = 0

                if draw_counter >= DRAW_CONSECUTIVE:
                    board.push(move)
                    ply += 1
                    termination = "draw_adjudication"
                    break

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
        matchup_name=matchup_name,
    )


# =========================================================
# Parallel worker
# =========================================================

def run_job(job: MatchJob) -> Tuple[MatchJob, List[GameResult]]:
    print(f"[worker] starting job {job.opening_idx}", flush=True)
    engine = None
    try:
        engine = open_engine()

        a_name, b_name = MATCHUP_REGISTRY[job.matchup_name]

        # Game 1: A as White
        a = make_policy(a_name, job.use_discrete)
        b = make_policy(b_name, job.use_discrete)
        g1 = play_one_game(
            engine=engine,
            opening_fen=job.opening_fen,
            white_policy=a,
            black_policy=b,
            opening_idx=job.opening_idx,
            matchup_name=job.matchup_name,
        )

        # Game 2: colour reversal
        a = make_policy(a_name, job.use_discrete)
        b = make_policy(b_name, job.use_discrete)
        g2 = play_one_game(
            engine=engine,
            opening_fen=job.opening_fen,
            white_policy=b,
            black_policy=a,
            opening_idx=job.opening_idx,
            matchup_name=job.matchup_name,
        )

        return job, [g1, g2]

    except Exception as e:
        print(
            f"[worker-error] matchup={job.matchup_name} opening_idx={job.opening_idx} "
            f"mode={'discrete' if job.use_discrete else 'continuous'} error={e}",
            flush=True,
        )
        return job, []

    finally:
        close_engine(engine)


# =========================================================
# Reporting / saving
# =========================================================

def summarise(results: List[GameResult], a_name: str):
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
    pct = (score / total * 100) if total else 0.0

    print(
        f"  {a_name}: +{a_w} ={d} -{a_l} (*{u})  "
        f"Score: {score}/{total} ({pct:.1f}%)"
    )

    terms: Dict[str, int] = {}
    for r in results:
        terms[r.termination] = terms.get(r.termination, 0) + 1
    print(f"  Terminations: {terms}")


def save_csv(results: List[GameResult], path: str):
    fields = [
        "opening_idx",
        "opening_fen",
        "white_policy",
        "black_policy",
        "discrete",
        "result",
        "termination",
        "total_plies",
        "white_budget_remaining",
        "black_budget_remaining",
        "white_nodes_total",
        "black_nodes_total",
        "matchup_name",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    print(f"  Saved: {path}")


# =========================================================
# Main
# =========================================================

def build_jobs(openings: List[str]) -> List[MatchJob]:
    jobs: List[MatchJob] = []
    for matchup_name in MATCHUP_REGISTRY:
        for use_discrete in [True, False]:
            for idx, fen in enumerate(openings):
                jobs.append(
                    MatchJob(
                        matchup_name=matchup_name,
                        opening_idx=idx,
                        opening_fen=fen,
                        use_discrete=use_discrete,
                    )
                )
    return jobs


def main():
    if not os.path.exists(OPENINGS_FILE):
        print(f"ERROR: {OPENINGS_FILE} not found.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found.")
        return

    with open(OPENINGS_FILE, "r", encoding="utf-8") as f:
        openings = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(openings)} openings")
    print(f"Using STOCKFISH_PATH={STOCKFISH_PATH}")
    print(f"Using MODEL_PATH={MODEL_PATH}")
    print(f"Using {NUM_WORKERS} workers")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs = build_jobs(openings)
    total_jobs = len(jobs)
    print(f"Built {total_jobs} jobs")

    grouped_results: Dict[str, Dict[bool, List[GameResult]]] = {
        matchup_name: {True: [], False: []}
        for matchup_name in MATCHUP_REGISTRY
    }

    t0 = time.time()

    with mp.Pool(NUM_WORKERS) as pool:
        for i, (job, pair_results) in enumerate(pool.imap_unordered(run_job, jobs), start=1):
            grouped_results[job.matchup_name][job.use_discrete].extend(pair_results)

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (total_jobs - i) / rate if rate > 0 else 0.0

            print(
                f"Completed {i}/{total_jobs} jobs | "
                f"Last finished: opening={job.opening_idx}, "
                f"matchup={job.matchup_name}, "
                f"mode={'discrete' if job.use_discrete else 'continuous'} | "
                f"Elapsed={elapsed/60:.1f} min | ETA={eta/60:.1f} min",
                flush=True,
            )

    elapsed = time.time() - t0

    for matchup_name, (a_name, _) in MATCHUP_REGISTRY.items():
        for use_discrete in [True, False]:
            mode = "discrete" if use_discrete else "continuous"
            results = grouped_results[matchup_name][use_discrete]

            print(f"\n{'='*60}")
            print(f"{matchup_name} ({mode}) — {len(results)} games")
            print(f"{'='*60}")
            summarise(results, a_name)

            csv_path = os.path.join(RESULTS_DIR, f"{matchup_name}_{mode}.csv")
            save_csv(results, csv_path)

    print(f"\n{'='*60}")
    print(f"Tournament complete in {elapsed/60:.1f} minutes")
    print(f"Results saved in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()