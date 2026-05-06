import random
import statistics
import time
from pathlib import Path
from typing import List, Dict, Any

import chess
import chess.engine

# =========================================================
# Config
# =========================================================

STOCKFISH_PATH = r"C:\Users\khoo\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
OPENINGS_FILE = "openings_100.txt"

THREADS = 1
HASH_MB = 128

# Use larger budgets to reduce Python/UCI call overhead distortion.
NODE_BUDGETS = [100_000, 400_000, 1_600_000]

# Use all 100 openings by default.
MAX_FENS = None
RANDOM_SEED = 42

# Warm-up runs are not included in final stats.
WARMUP_FENS = 5

# Repeat the whole benchmark
REPEATS = 3

# Trim fraction for trimmed mean (e.g. 0.1 = drop top/bottom 10%)
TRIM_FRAC = 0.10


# =========================================================
# Helpers
# =========================================================

def load_fens(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        fens = [line.strip() for line in f if line.strip()]
    return fens


def open_engine() -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({
        "Threads": THREADS,
        "Hash": HASH_MB,
    })
    return engine


def trimmed_mean(values: List[float], trim_frac: float = 0.10) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    k = int(len(vals) * trim_frac)
    if len(vals) - 2 * k <= 0:
        return statistics.mean(vals)
    vals = vals[k: len(vals) - k]
    return statistics.mean(vals)


def run_single_analysis(
    engine: chess.engine.SimpleEngine,
    fen: str,
    nodes: int
) -> Dict[str, Any]:
    board = chess.Board(fen)

    t0 = time.perf_counter()
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    elapsed = time.perf_counter() - t0

    actual_nodes = int(info.get("nodes", nodes))
    nps = actual_nodes / elapsed if elapsed > 0 else float("nan")

    return {
        "fen": fen,
        "budget_nodes": nodes,
        "actual_nodes": actual_nodes,
        "elapsed_s": elapsed,
        "nps": nps,
        "depth": info.get("depth"),
        "seldepth": info.get("seldepth"),
    }


# =========================================================
# Main benchmark
# =========================================================

def main() -> None:
    random.seed(RANDOM_SEED)

    fens = load_fens(OPENINGS_FILE)
    if MAX_FENS is not None:
        fens = fens[:MAX_FENS]

    random.shuffle(fens)

    print(f"Loaded {len(fens)} FENs from {OPENINGS_FILE}")
    print(f"Budgets: {NODE_BUDGETS}")
    print(f"Threads={THREADS}, Hash={HASH_MB} MB, Repeats={REPEATS}")
    print()

    all_rows: List[Dict[str, Any]] = []

    engine = open_engine()
    try:
        # -------------------------
        # Warm-up
        # -------------------------
        print("Warm-up phase...")
        warmup_positions = fens[:min(WARMUP_FENS, len(fens))]
        for fen in warmup_positions:
            _ = run_single_analysis(engine, fen, NODE_BUDGETS[0])
        print("Warm-up complete.\n")

        # -------------------------
        # Benchmark
        # -------------------------
        for rep in range(REPEATS):
            print(f"Repeat {rep + 1}/{REPEATS}")

            # Shuffle order each repeat to reduce positional ordering bias
            rep_fens = fens[:]
            random.shuffle(rep_fens)

            for budget in NODE_BUDGETS:
                print(f"  Budget {budget} nodes...")

                for i, fen in enumerate(rep_fens, start=1):
                    row = run_single_analysis(engine, fen, budget)
                    row["repeat"] = rep + 1
                    all_rows.append(row)

                    if i % 20 == 0 or i == len(rep_fens):
                        print(f"    {i}/{len(rep_fens)}")

            print()

    finally:
        try:
            engine.quit()
        except Exception:
            pass

    # =====================================================
    # Summaries
    # =====================================================

    print("\n==================== SUMMARY ====================\n")

    def summarise_subset(rows: List[Dict[str, Any]], label: str) -> None:
        nps_values = [r["nps"] for r in rows]
        total_nodes = sum(r["actual_nodes"] for r in rows)
        total_time = sum(r["elapsed_s"] for r in rows)

        weighted_overall_nps = total_nodes / total_time if total_time > 0 else float("nan")
        median_nps = statistics.median(nps_values)
        mean_nps = statistics.mean(nps_values)
        tmean_nps = trimmed_mean(nps_values, TRIM_FRAC)

        print(label)
        print(f"  Samples                : {len(rows)}")
        print(f"  Total nodes            : {total_nodes:,}")
        print(f"  Total elapsed time (s) : {total_time:.4f}")
        print(f"  Weighted overall NPS   : {weighted_overall_nps:,.0f}")
        print(f"  Median NPS             : {median_nps:,.0f}")
        print(f"  Mean NPS               : {mean_nps:,.0f}")
        print(f"  Trimmed mean NPS       : {tmean_nps:,.0f}")
        print()

    # Per-budget summary
    for budget in NODE_BUDGETS:
        rows = [r for r in all_rows if r["budget_nodes"] == budget]
        summarise_subset(rows, f"Budget = {budget:,} nodes")

    # Overall summary
    summarise_subset(all_rows, "Overall across all budgets")

    # Optional: print a recommendation
    total_nodes = sum(r["actual_nodes"] for r in all_rows)
    total_time = sum(r["elapsed_s"] for r in all_rows)
    recommended_nps = total_nodes / total_time if total_time > 0 else float("nan")

    print("Final NPS:")
    print(f"  {recommended_nps:,.0f} nodes/s")
    print("\nUse the weighted overall NPS above as the main estimate.")


if __name__ == "__main__":
    main()