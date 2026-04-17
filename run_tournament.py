"""
run_tournament.py
=================

Parallel tournament runner with per-side TT isolation.

Policies tested:
  1. FixedUniform
  2. SolakVuckovic
  3. Hyatt
  4. TokenBucket

Supports:
  - static-only models
  - probe models

IMPORTANT TT DESIGN
-------------------
1) Each side gets its own Stockfish engine instance per game.
2) White and Black do NOT share a TT.
3) Engines are restarted between games, so TT does NOT carry across games/matchups.
4)For TokenBucket with probe features, TT persists across 2k, 5k and actual search.
"""

import csv
import os
import time
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from collections import Counter

import chess
import chess.engine
import joblib
import pandas as pd


# =========================================================
# Config
# =========================================================

STOCKFISH_PATH = "stockfish"
MODEL_PATH = "trained_models/gbt_probe_tol20.joblib"
OPENINGS_FILE = "openings_100.txt"
RESULTS_DIR = "tournament_results_probe"

TOTAL_NODE_BUDGET = 10_000_000
BUCKETS = [25_000, 100_000, 400_000, 1_600_000]

SMALL_PROBE_NODES = 2_000
LARGE_PROBE_NODES = 5_000

# Initial estimate of remaining side-moves in a game
INITIAL_MOVES_ESTIMATE = 64

# Quick-run control
NUM_OPENINGS = 5          # None = use all openings

# Debug
DEBUG_CONTROLLER = True
DEBUG_MAX_RECORDS_PER_JOB = 50
DEBUG_PRINT_FIRST_N = 12

# Adjudication
MAX_PLIES = 400
DRAW_MOVE_THRESHOLD = 40
DRAW_CP_THRESHOLD = 5
DRAW_CONSECUTIVE = 10
RESIGN_CP_THRESHOLD = 1000
RESIGN_CONSECUTIVE = 5

HASH_MB = 128
THREADS = 1

# Assuming 16 vCPUs like I am using in Hetzner
NUM_WORKERS = 8


# =========================================================
# Engine helpers
# =========================================================

def open_engine() -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({
        "Threads": THREADS,
        "Hash": HASH_MB,
    })
    return engine


def close_engine(engine: Optional[chess.engine.SimpleEngine]) -> None:
    if engine is not None:
        try:
            engine.quit()
        except Exception:
            pass


# =========================================================
# Static feature extraction
# =========================================================

def extract_static_features(board: chess.Board) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}

    num_legal_moves = board.legal_moves.count()
    feats["num_legal_moves"] = num_legal_moves
    feats["is_check"] = int(board.is_check())
    feats["side_to_move_white"] = int(board.turn == chess.WHITE)

    num_captures = 0
    num_checks = 0
    num_promotions = 0

    for move in board.legal_moves:
        if board.is_capture(move):
            num_captures += 1
        if move.promotion is not None:
            num_promotions += 1
        if board.gives_check(move):
            num_checks += 1

    denom = max(1, num_legal_moves)
    feats["capture_ratio"] = num_captures / denom
    feats["check_ratio"] = num_checks / denom
    feats["num_promotions"] = num_promotions

    occupied_by_us = board.occupied_co[board.turn]

    mobility_map = {
        chess.KNIGHT: "knight_mobility",
        chess.BISHOP: "bishop_mobility",
        chess.ROOK: "rook_mobility",
        chess.QUEEN: "queen_mobility",
    }

    for piece_type, feat_name in mobility_map.items():
        total = 0
        for sq in board.pieces(piece_type, board.turn):
            total += len(board.attacks(sq) & ~occupied_by_us)
        feats[feat_name] = total

    enemy_king_sq = board.king(not board.turn)
    feats["num_attackers_on_king"] = (
        len(board.attackers(board.turn, enemy_king_sq)) if enemy_king_sq is not None else 0
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


def score_to_cp(
    score_obj: Optional[chess.engine.PovScore]
) -> Optional[int]:
    if score_obj is None:
        return None
    white_score = score_obj.white()
    if white_score.is_mate():
        return None
    return white_score.score()


def sign_of_cp(x: Optional[int]) -> int:
    if x is None:
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def unpack_probe_infos(infos) -> Tuple[Dict, Dict]:
    if isinstance(infos, list):
        pv1 = infos[0] if len(infos) >= 1 else {}
        pv2 = infos[1] if len(infos) >= 2 else {}
    else:
        pv1 = infos
        pv2 = {}
    return pv1, pv2


def extract_top_move_uci(pv_info: Dict) -> Optional[str]:
    pv = pv_info.get("pv", [])
    return pv[0].uci() if pv else None


def best_second_gap_cp(pv1: Dict, pv2: Dict) -> int:
    cp1 = score_to_cp(pv1.get("score"))
    cp2 = score_to_cp(pv2.get("score")) if pv2 else None
    if cp1 is not None and cp2 is not None:
        return cp1 - cp2
    return 0


def extract_probe_features(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    small_probe_nodes: int = SMALL_PROBE_NODES,
    large_probe_nodes: int = LARGE_PROBE_NODES,
) -> Dict[str, Any]:
    """
    Uses the SAME engine for:
      2k probe -> 5k probe
    so TT is shared across those probe stages, and then the actual search
    also uses the same engine in play_one_game().
    """
    feats: Dict[str, Any] = {}

    infos_small = engine.analyse(
        board,
        chess.engine.Limit(nodes=small_probe_nodes),
        multipv=2
    )

    infos_large = engine.analyse(
        board,
        chess.engine.Limit(nodes=large_probe_nodes),
        multipv=2
    )

    small_pv1, small_pv2 = unpack_probe_infos(infos_small)
    large_pv1, large_pv2 = unpack_probe_infos(infos_large)

    # Keep
    large_score_type, large_score_val = extract_score_info(large_pv1.get("score"))
    feats["probe_score_cp"] = large_score_val if large_score_type == "cp" else None
    feats["probe_abs_score_cp"] = abs(large_score_val) if large_score_type == "cp" and large_score_val is not None else None
    feats["probe_is_mate"] = int(large_score_type == "mate")
    feats["probe_depth"] = large_pv1.get("depth")
    feats["probe_seldepth"] = large_pv1.get("seldepth")
    feats["probe_best_second_gap"] = best_second_gap_cp(large_pv1, large_pv2)

    # Add
    small_cp = score_to_cp(small_pv1.get("score"))
    large_cp = score_to_cp(large_pv1.get("score"))

    small_gap = best_second_gap_cp(small_pv1, small_pv2)
    large_gap = feats["probe_best_second_gap"]

    small_top_move = extract_top_move_uci(small_pv1)
    large_top_move = extract_top_move_uci(large_pv1)

    feats["probe_score_delta_small"] = abs(large_cp - small_cp) if small_cp is not None and large_cp is not None else 0
    feats["probe_gap_delta_small"] = abs(large_gap - small_gap)
    feats["probe_sign_flip"] = int(sign_of_cp(small_cp) != sign_of_cp(large_cp))
    feats["probe_top_move_changed"] = int(
        small_top_move is not None and large_top_move is not None and small_top_move != large_top_move
    )

    return feats


# =========================================================
# Moves-left estimate
# =========================================================

def estimate_moves_left(ply: int) -> int:
    """
    Simple decreasing side-move estimate starting from INITIAL_MOVES_ESTIMATE.
    """
    side_moves_played = ply // 2
    return max(1, INITIAL_MOVES_ESTIMATE - side_moves_played)


# =========================================================
# Bucket helper
# =========================================================

def snap_to_bucket(nodes: float) -> int:
    affordable = [b for b in BUCKETS if b <= nodes]
    if affordable:
        return max(affordable)
    return min(int(nodes), BUCKETS[0])


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
    def decide_nodes(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board,
        ply: int
    ) -> int:
        raise NotImplementedError

    def consume(self, nodes_used: int):
        self.remaining_budget = max(0, self.remaining_budget - nodes_used)

    def finalise(self, raw: float) -> int:
        raw = min(raw, self.remaining_budget)
        if self.use_discrete:
            raw = max(BUCKETS[0], raw)
            return snap_to_bucket(raw)
        return max(1, int(raw))

    def reset(self):
        self.remaining_budget = self.total_budget


class FixedUniformPolicy(AllocationPolicy):
    def __init__(self, total_budget: int, use_discrete: bool):
        super().__init__("FixedUniform", total_budget, use_discrete)
        self.fixed_allocation = total_budget / INITIAL_MOVES_ESTIMATE

    def decide_nodes(self, engine, board, ply) -> int:
        raw = min(self.fixed_allocation, self.remaining_budget)
        return self.finalise(raw)


class SolakVuckovicPolicy(AllocationPolicy):
    def __init__(self, total_budget: int, use_discrete: bool):
        super().__init__("SolakVuckovic", total_budget, use_discrete)

    def decide_nodes(self, engine, board, ply) -> int:
        moves_left = estimate_moves_left(ply)
        raw = self.remaining_budget / moves_left
        return self.finalise(raw)


class HyattPolicy(AllocationPolicy):
    def __init__(self, total_budget: int, use_discrete: bool):
        super().__init__("Hyatt", total_budget, use_discrete)

    def decide_nodes(self, engine, board, ply) -> int:
        moves_left = estimate_moves_left(ply)
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
        debug_rows: Optional[List[Dict[str, Any]]] = None,
        burst_cap: int = BUCKETS[-1] * 2
    ):
        super().__init__("TokenBucket", total_budget, use_discrete)

        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_cols = payload["feature_cols"]
        self.uses_probe = any(col.startswith("probe_") for col in self.feature_cols)

        self.burst_cap = burst_cap
        self.tokens = float(burst_cap)
        self.refill_rate = total_budget / INITIAL_MOVES_ESTIMATE

        self.debug_source_tag = "unset"
        self.debug_rows = debug_rows if debug_rows is not None else []

    def reset(self):
        super().reset()
        self.tokens = float(self.burst_cap)
        self.refill_rate = self.total_budget / INITIAL_MOVES_ESTIMATE

    def set_debug_source_tag(self, tag: str):
        self.debug_source_tag = tag

    def add_debug_row(self, row: Dict[str, Any]) -> None:
        if DEBUG_CONTROLLER and len(self.debug_rows) < DEBUG_MAX_RECORDS_PER_JOB:
            self.debug_rows.append(row)

    def build_feature_payload(
        self,
        engine: chess.engine.SimpleEngine,
        board: chess.Board
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        feats = extract_static_features(board)

        if self.uses_probe:
            feats.update(extract_probe_features(engine, board))

        row: Dict[str, Any] = {}
        raw_view: Dict[str, Any] = {}

        for col in self.feature_cols:
            original_val = feats.get(col, 0)
            raw_view[col] = original_val

            val = original_val
            if val is None:
                val = 0
            elif isinstance(val, str):
                val = 0

            row[col] = val

        X = pd.DataFrame([row], columns=self.feature_cols)
        return X, row, raw_view

    def decide_nodes(self, engine, board, ply) -> int:
        moves_left = estimate_moves_left(ply)
        self.refill_rate = self.remaining_budget / moves_left

        X, sanitized_row, raw_view = self.build_feature_payload(engine, board)
        predicted = int(self.model.predict(X)[0])

        spendable = min(predicted, self.tokens, self.remaining_budget)
        final_nodes = self.finalise(spendable)

        interesting_cols = [
            "num_legal_moves",
            "is_check",
            "capture_ratio",
            "check_ratio",
            "num_promotions",
            "knight_mobility",
            "bishop_mobility",
            "rook_mobility",
            "queen_mobility",
            "num_attackers_on_king",
            "num_pinned_pieces",
            "side_to_move_white",
            "probe_score_cp",
            "probe_abs_score_cp",
            "probe_best_second_gap",
            "probe_depth",
            "probe_seldepth",
            "probe_is_mate",
            "probe_score_delta_small",
            "probe_gap_delta_small",
            "probe_sign_flip",
            "probe_top_move_changed",
        ]

        feature_snapshot = {k: raw_view[k] for k in interesting_cols if k in raw_view}
        sanitized_snapshot = {k: sanitized_row[k] for k in interesting_cols if k in sanitized_row}

        self.add_debug_row({
            "source_tag": self.debug_source_tag,
            "ply": ply,
            "turn_white": int(board.turn == chess.WHITE),
            "fen": board.fen(),
            "uses_probe": self.uses_probe,
            "use_discrete": self.use_discrete,
            "moves_left_estimate": moves_left,
            "remaining_budget_before": self.remaining_budget,
            "tokens_before": self.tokens,
            "predicted_raw": predicted,
            "spendable_before_finalise": spendable,
            "final_nodes": final_nodes,
            "feature_snapshot_raw": feature_snapshot,
            "feature_snapshot_sanitized": sanitized_snapshot,
        })

        return final_nodes

    def consume(self, nodes_used: int):
        super().consume(nodes_used)
        self.tokens = max(0.0, self.tokens - nodes_used)
        self.tokens = min(self.tokens + self.refill_rate, float(self.burst_cap))


# =========================================================
# Results / jobs
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


@dataclass
class MatchJob:
    matchup_name: str
    opening_idx: int
    opening_fen: str
    use_discrete: bool


MATCHUP_REGISTRY = {
    "TokenBucket_vs_FixedUniform": ("TokenBucket", "FixedUniform"),
    "TokenBucket_vs_SolakVuckovic": ("TokenBucket", "SolakVuckovic"),
    "TokenBucket_vs_Hyatt": ("TokenBucket", "Hyatt"),
}


# =========================================================
# Policy factory
# =========================================================

def make_policy(
    policy_name: str,
    use_discrete: bool,
    debug_rows: Optional[List[Dict[str, Any]]] = None
) -> AllocationPolicy:
    if policy_name == "TokenBucket":
        return TokenBucketPolicy(MODEL_PATH, TOTAL_NODE_BUDGET, use_discrete, debug_rows=debug_rows)
    if policy_name == "FixedUniform":
        return FixedUniformPolicy(TOTAL_NODE_BUDGET, use_discrete)
    if policy_name == "SolakVuckovic":
        return SolakVuckovicPolicy(TOTAL_NODE_BUDGET, use_discrete)
    if policy_name == "Hyatt":
        return HyattPolicy(TOTAL_NODE_BUDGET, use_discrete)
    raise ValueError(f"Unknown policy: {policy_name}")


# =========================================================
# Game runner
# =========================================================

def play_one_game(
    white_engine: chess.engine.SimpleEngine,
    black_engine: chess.engine.SimpleEngine,
    opening_fen: str,
    white_policy: AllocationPolicy,
    black_policy: AllocationPolicy,
    opening_idx: int,
) -> GameResult:
    board = chess.Board(opening_fen)
    white_policy.reset()
    black_policy.reset()

    if isinstance(white_policy, TokenBucketPolicy):
        white_policy.set_debug_source_tag(f"opening{opening_idx}_white_{white_policy.name}")
    if isinstance(black_policy, TokenBucketPolicy):
        black_policy.set_debug_source_tag(f"opening{opening_idx}_black_{black_policy.name}")

    w_nodes = 0
    b_nodes = 0
    draw_counter = 0
    resign_w = 0
    resign_b = 0
    ply = 0
    termination = "max_plies"
    budget_forfeit_side = None

    while not board.is_game_over() and ply < MAX_PLIES:
        if board.turn == chess.WHITE:
            engine = white_engine
            policy = white_policy
        else:
            engine = black_engine
            policy = black_policy

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
# Worker
# =========================================================

def run_job(job: MatchJob) -> Tuple[MatchJob, List[GameResult], List[Dict[str, Any]]]:
    debug_rows: List[Dict[str, Any]] = []
    results: List[GameResult] = []

    a_name, b_name = MATCHUP_REGISTRY[job.matchup_name]

    try:
        # Game 1: A as White
        white_engine = None
        black_engine = None
        try:
            white_engine = open_engine()
            black_engine = open_engine()

            a = make_policy(a_name, job.use_discrete, debug_rows=debug_rows)
            b = make_policy(b_name, job.use_discrete, debug_rows=debug_rows)
            g1 = play_one_game(white_engine, black_engine, job.opening_fen, a, b, job.opening_idx)
            results.append(g1)
        finally:
            close_engine(white_engine)
            close_engine(black_engine)

        # Game 2: B as White
        white_engine = None
        black_engine = None
        try:
            white_engine = open_engine()
            black_engine = open_engine()

            a = make_policy(a_name, job.use_discrete, debug_rows=debug_rows)
            b = make_policy(b_name, job.use_discrete, debug_rows=debug_rows)
            g2 = play_one_game(white_engine, black_engine, job.opening_fen, b, a, job.opening_idx)
            results.append(g2)
        finally:
            close_engine(white_engine)
            close_engine(black_engine)

    except Exception as e:
        print(
            f"[worker-error] opening={job.opening_idx} matchup={job.matchup_name} "
            f"mode={'discrete' if job.use_discrete else 'continuous'} error={e}",
            flush=True,
        )

    return job, results, debug_rows


# =========================================================
# Saving / reporting
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
    pct = score / total * 100 if total else 0

    print(f"\n  {a_name}: +{a_w} ={d} -{a_l} (*{u})  Score: {score}/{total} ({pct:.1f}%)")

    terms = {}
    for r in results:
        terms[r.termination] = terms.get(r.termination, 0) + 1
    print(f"  Terminations: {terms}")


def save_csv(results: List[GameResult], path: str):
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
            w.writerow(asdict(r))
    print(f"  Saved: {path}")


def print_debug_summary(debug_rows: List[Dict[str, Any]]):
    if not DEBUG_CONTROLLER:
        return

    print(f"\n{'='*60}")
    print("CONTROLLER DEBUG SUMMARY")
    print(f"{'='*60}")

    if not debug_rows:
        print("No debug rows captured.")
        return

    print(f"Captured debug rows: {len(debug_rows)}")

    predicted_counter = Counter(row["predicted_raw"] for row in debug_rows)
    final_counter = Counter(row["final_nodes"] for row in debug_rows)

    print("\nPredicted raw values frequency:")
    for k, v in predicted_counter.most_common():
        print(f"  {k}: {v}")

    print("\nFinal allocated nodes frequency:")
    for k, v in final_counter.most_common():
        print(f"  {k}: {v}")

    avg_pred = sum(row["predicted_raw"] for row in debug_rows) / len(debug_rows)
    avg_final = sum(row["final_nodes"] for row in debug_rows) / len(debug_rows)
    avg_spendable = sum(row["spendable_before_finalise"] for row in debug_rows) / len(debug_rows)

    print(f"\nAverage predicted_raw: {avg_pred:.2f}")
    print(f"Average spendable_before_finalise: {avg_spendable:.2f}")
    print(f"Average final_nodes: {avg_final:.2f}")

    unexpected_preds = [
        row["predicted_raw"] for row in debug_rows
        if row["use_discrete"] and row["predicted_raw"] not in BUCKETS
    ]
    if unexpected_preds:
        print("\nWARNING: In discrete mode, some raw predictions are not one of the bucket values.")
        print(f"Example unexpected predictions: {unexpected_preds[:10]}")

    print(f"\nFirst {min(DEBUG_PRINT_FIRST_N, len(debug_rows))} debug rows:")
    for i, row in enumerate(debug_rows[:DEBUG_PRINT_FIRST_N], start=1):
        print(f"\n--- Debug row {i} ---")
        print(f"source_tag                : {row['source_tag']}")
        print(f"ply                       : {row['ply']}")
        print(f"turn_white                : {row['turn_white']}")
        print(f"uses_probe                : {row['uses_probe']}")
        print(f"use_discrete              : {row['use_discrete']}")
        print(f"moves_left_estimate       : {row['moves_left_estimate']}")
        print(f"remaining_budget_before   : {row['remaining_budget_before']}")
        print(f"tokens_before             : {row['tokens_before']:.2f}")
        print(f"predicted_raw             : {row['predicted_raw']}")
        print(f"spendable_before_finalise : {row['spendable_before_finalise']}")
        print(f"final_nodes               : {row['final_nodes']}")
        print(f"feature_snapshot_raw      : {row['feature_snapshot_raw']}")
        print(f"feature_snapshot_sanitized: {row['feature_snapshot_sanitized']}")
        print(f"fen                       : {row['fen']}")


# =========================================================
# Main
# =========================================================

def build_jobs(openings: List[str]) -> List[MatchJob]:
    jobs: List[MatchJob] = []
    for matchup_name in MATCHUP_REGISTRY:
        for use_discrete in [True, False]:
            for idx, fen in enumerate(openings):
                jobs.append(MatchJob(
                    matchup_name=matchup_name,
                    opening_idx=idx,
                    opening_fen=fen,
                    use_discrete=use_discrete,
                ))
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

    if NUM_OPENINGS is not None:
        openings = openings[:NUM_OPENINGS]

    print(f"Loaded {len(openings)} openings")
    print(f"MODEL_PATH={MODEL_PATH}")
    print(f"NUM_OPENINGS={NUM_OPENINGS}")
    print(f"NUM_WORKERS={NUM_WORKERS}")
    print(f"DEBUG_CONTROLLER={DEBUG_CONTROLLER}")
    print(f"INITIAL_MOVES_ESTIMATE={INITIAL_MOVES_ESTIMATE}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs = build_jobs(openings)
    total_jobs = len(jobs)
    print(f"Built {total_jobs} jobs")

    grouped_results: Dict[str, Dict[bool, List[GameResult]]] = {
        matchup_name: {True: [], False: []}
        for matchup_name in MATCHUP_REGISTRY
    }
    all_debug_rows: List[Dict[str, Any]] = []

    t0 = time.time()

    with mp.Pool(NUM_WORKERS) as pool:
        for i, (job, job_results, job_debug_rows) in enumerate(pool.imap_unordered(run_job, jobs), start=1):
            grouped_results[job.matchup_name][job.use_discrete].extend(job_results)
            if DEBUG_CONTROLLER:
                all_debug_rows.extend(job_debug_rows)

            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (total_jobs - i) / rate if rate > 0 else 0.0

            print(
                f"Completed {i}/{total_jobs} jobs | "
                f"Last: opening={job.opening_idx}, matchup={job.matchup_name}, "
                f"mode={'discrete' if job.use_discrete else 'continuous'} | "
                f"Elapsed={elapsed/60:.1f} min | ETA={eta/60:.1f} min",
                flush=True,
            )

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

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Tournament complete in {elapsed/60:.1f} minutes")
    print(f"Results in {RESULTS_DIR}/")

    print_debug_summary(all_debug_rows)


if __name__ == "__main__":
    # On Linux/Hetzner, default start method is usually fine.
    main()