"""
generate_openings.py
====================
Generates a stratified set of 100 opening positions for engine-vs-engine testing.
No external PGN database required — positions are generated from scratch
using Stockfish MultiPV with controlled randomisation.

Usage:
  python generate_openings.py

Output:
  openings_100.txt            — one FEN per line
  openings_100_annotated.txt  — FEN + eval + move sequence
"""

import chess
import chess.engine
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass

# =========================================================
# Config
# =========================================================

STOCKFISH_PATH = r"C:\Users\khoo\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
OUTPUT_FILE = "openings_100.txt"
SEED = 42

# Ply depth at which to extract positions.
EXTRACT_PLY = 16

# Generation: MultiPV breadth and shallow search budget per ply.
MULTI_PV = 5
GEN_NODES = 10_000

# How many unique lines to generate before filtering.
NUM_LINES = 1500

# Evaluation: deeper search for accurate band assignment.
EVAL_NODES = 10_000_000

# Eval bands: (min_abs_cp, max_abs_cp, num_to_sample)
EVAL_BANDS = [
    (0,   30,  40),   # Near-equal eval
    (30,  70,  40),   # Moderately imbalanced
    (70,  120, 20),   # Clearly imbalanced
]


# =========================================================
# Data
# =========================================================

@dataclass
class CandidateOpening:
    fen: str
    eval_cp: int
    abs_eval_cp: int
    move_sequence: str


# =========================================================
# Helpers
# =========================================================

def pick_move_weighted(infos: list, rng: random.Random, temperature: float) -> Optional[chess.Move]:
    """Pick a move from MultiPV results with rank-based weighting."""
    if not infos:
        return None

    weights = [(1.0 / temperature) ** i for i in range(len(infos))]
    total = sum(weights)
    weights = [w / total for w in weights]

    idx = rng.choices(range(len(infos)), weights=weights, k=1)[0]
    pv = infos[idx].get("pv", [])
    return pv[0] if pv else None


def generate_one_line(engine, rng) -> Optional[Tuple[str, str]]:
    """
    Play one opening line with randomised MultiPV move selection.
    Returns (fen, move_sequence_string) or None.
    """
    board = chess.Board()
    moves_played = []

    for ply in range(EXTRACT_PLY):
        if board.is_game_over():
            return None

        try:
            infos = engine.analyse(board, chess.engine.Limit(nodes=GEN_NODES), multipv=MULTI_PV)
        except Exception:
            return None

        if not isinstance(infos, list):
            infos = [infos]

        valid = [info for info in infos if info.get("pv") and info["pv"][0] in board.legal_moves]
        if not valid:
            return None

        # More random early (explore openings), more deterministic later
        if ply < 6:
            temp = 2.0
        elif ply < 12:
            temp = 1.5
        else:
            temp = 1.2

        move = pick_move_weighted(valid, rng, temp)
        if move is None or move not in board.legal_moves:
            return None

        moves_played.append(board.san(move))
        board.push(move)

    if board.is_game_over():
        return None

    parts = []
    for i, san in enumerate(moves_played):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(san)

    return board.fen(), " ".join(parts)


def evaluate_fen(engine, fen: str) -> Optional[int]:
    """Return eval in centipawns from white's POV, or None for mates."""
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(nodes=EVAL_NODES))
    score = info.get("score")
    if score is None:
        return None
    white_score = score.white()
    if white_score.is_mate():
        return None
    return white_score.score()


# =========================================================
# Main
# =========================================================

def main():
    rng = random.Random(SEED)

    # Phase 1: Generate diverse opening lines

    print(f"Phase 1: Generating up to {NUM_LINES} unique opening lines...")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1, "Hash": 32})

    candidates = []
    seen = set()
    attempts = 0

    try:
        while len(candidates) < NUM_LINES:
            attempts += 1

            result = generate_one_line(engine, rng)
            if result is None:
                continue

            fen, moves = result
            fen_key = " ".join(fen.split()[:4])

            if fen_key in seen:
                continue
            seen.add(fen_key)

            candidates.append({"fen": fen, "moves": moves})

            if len(candidates) % 100 == 0:
                print(f"  {len(candidates)}/{NUM_LINES} unique ({attempts} attempts)")

            if attempts > NUM_LINES * 10:
                print(f"  Stopping at {attempts} attempts (diversity limit)")
                break
    finally:
        engine.quit()

    print(f"  Done: {len(candidates)} unique positions from {attempts} attempts\n")

    # Phase 2: Evaluate positions for band assignment

    max_abs = max(b[1] for b in EVAL_BANDS)
    print(f"Phase 2: Evaluating {len(candidates)} positions ({EVAL_NODES} nodes each)...")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1, "Hash": 64})

    evaluated = []
    try:
        for i, cand in enumerate(candidates, 1):
            cp = evaluate_fen(engine, cand["fen"])

            if cp is not None and abs(cp) <= max_abs:
                evaluated.append(CandidateOpening(
                    fen=cand["fen"],
                    eval_cp=cp,
                    abs_eval_cp=abs(cp),
                    move_sequence=cand["moves"],
                ))

            if i % 200 == 0:
                print(f"  Evaluated {i}/{len(candidates)}, kept {len(evaluated)}")
    finally:
        engine.quit()

    print(f"  Kept {len(evaluated)} positions within [0, {max_abs}]cp\n")

    # Phase 3: Stratified sampling

    print("Phase 3: Stratified sampling...")
    selected = []

    for min_cp, max_cp, n_sample in EVAL_BANDS:
        band = [p for p in evaluated if min_cp <= p.abs_eval_cp < max_cp]
        rng.shuffle(band)

        take = min(n_sample, len(band))
        selected.extend(band[:take])

        status = "OK" if take >= n_sample else "SHORT"
        print(f"  [{min_cp:>3d}, {max_cp:>3d})cp: {len(band):>4d} available, "
              f"sampled {take:>3d}/{n_sample}  [{status}]")

    rng.shuffle(selected)
    print(f"  Total: {len(selected)} positions\n")

    # Phase 4: Save

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pos in selected:
            f.write(pos.fen + "\n")

    annotated = OUTPUT_FILE.replace(".txt", "_annotated.txt")
    with open(annotated, "w", encoding="utf-8") as f:
        f.write(f"{'#':<4s} {'Eval':>6s}  {'FEN':<80s}  Moves\n")
        f.write("=" * 140 + "\n")
        for i, pos in enumerate(selected, 1):
            sign = "+" if pos.eval_cp >= 0 else ""
            f.write(f"{i:<4d} {sign}{pos.eval_cp:>5d}  "
                    f"{pos.fen:<80s}  {pos.move_sequence}\n")

    # Summary
    evals = [p.eval_cp for p in selected]
    abs_evals = [p.abs_eval_cp for p in selected]
    w_fav = sum(1 for e in evals if e > 10)
    b_fav = sum(1 for e in evals if e < -10)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"       {annotated}")
    print(f"\nEval range:    {min(evals):+d} to {max(evals):+d} cp")
    print(f"Mean |eval|:   {sum(abs_evals)/len(abs_evals):.1f} cp")
    print(f"White-favoured (>+10cp): {w_fav}")
    print(f"Black-favoured (<-10cp): {b_fav}")
    print(f"Balanced (±10cp):        {len(evals) - w_fav - b_fav}")


if __name__ == "__main__":
    main()