import joblib
import chess
import chess.engine
import chess.pgn
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

# =========================================================
# Config
# =========================================================

STOCKFISH_PATH = r"C:\Users\khoo\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
MODEL_PATH = "trained_models/gbt_static_tol20.joblib"

BUCKETS = [25_000, 100_000, 400_000, 1_600_000]
DEFAULT_TOTAL_NODE_BUDGET = 10_000_000

# Token bucket parameters
# Burst cap: maximum tokens that can accumulate 
DEFAULT_BURST_CAP = BUCKETS[-1] * 2

# Initial game length estimate used to seed the first refill rate.
# Recalculated dynamically each move based on piece count.
DEFAULT_EXPECTED_GAME_LENGTH = 50


# =========================================================
# Static feature extraction
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


def get_piece_counts(board: chess.Board) -> Dict[str, int]:
    feats = {}
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        name = PIECE_NAMES[piece_type]
        feats[f"white_{name}_count"] = len(board.pieces(piece_type, chess.WHITE))
        feats[f"black_{name}_count"] = len(board.pieces(piece_type, chess.BLACK))
    return feats


def get_total_material_and_imbalance(board: chess.Board):
    total_material = 0
    material_imbalance = 0

    for piece_type, value in PIECE_VALUES.items():
        w = len(board.pieces(piece_type, chess.WHITE))
        b = len(board.pieces(piece_type, chess.BLACK))
        total_material += value * (w + b)
        material_imbalance += value * (w - b)

    return total_material, material_imbalance


def count_legal_move_types(board: chess.Board):
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

    return num_captures, num_checks, num_promotions


def mobility_by_piece_type(board: chess.Board, color: chess.Color):
    result = {
        "knight_mobility": 0,
        "bishop_mobility": 0,
        "rook_mobility": 0,
        "queen_mobility": 0,
    }

    mapping = {
        chess.KNIGHT: "knight_mobility",
        chess.BISHOP: "bishop_mobility",
        chess.ROOK: "rook_mobility",
        chess.QUEEN: "queen_mobility",
    }

    occupied_by_us = board.occupied_co[color]

    for piece_type, feat_name in mapping.items():
        total = 0
        for sq in board.pieces(piece_type, color):
            attacks = board.attacks(sq)
            total += len(attacks & ~occupied_by_us)
        result[feat_name] = total

    return result


def num_attackers_on_enemy_king(board: chess.Board) -> int:
    stm = board.turn
    enemy = not stm
    enemy_king_sq = board.king(enemy)
    if enemy_king_sq is None:
        return 0
    return len(board.attackers(stm, enemy_king_sq))


def num_pinned_pieces(board: chess.Board, color: chess.Color) -> int:
    count = 0
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type != chess.KING:
            if board.is_pinned(color, sq):
                count += 1
    return count


def extract_static_features(board: chess.Board) -> Dict[str, int]:
    feats = {}

    feats["num_legal_moves"] = board.legal_moves.count()
    feats["is_check"] = int(board.is_check())

    feats.update(get_piece_counts(board))

    total_material, material_imbalance = get_total_material_and_imbalance(board)
    feats["total_material"] = total_material
    feats["material_imbalance"] = material_imbalance

    feats["total_pieces"] = sum(1 for p in board.piece_map().values() if p.piece_type != chess.KING)

    feats["side_to_move_white"] = int(board.turn == chess.WHITE)

    feats["white_can_castle_kingside"] = int(board.has_kingside_castling_rights(chess.WHITE))
    feats["white_can_castle_queenside"] = int(board.has_queenside_castling_rights(chess.WHITE))
    feats["black_can_castle_kingside"] = int(board.has_kingside_castling_rights(chess.BLACK))
    feats["black_can_castle_queenside"] = int(board.has_queenside_castling_rights(chess.BLACK))

    feats["halfmove_clock"] = board.halfmove_clock

    num_captures, num_checks, num_promotions = count_legal_move_types(board)
    feats["num_captures"] = num_captures
    feats["num_checks"] = num_checks
    feats["num_promotions"] = num_promotions

    feats.update(mobility_by_piece_type(board, board.turn))

    feats["num_attackers_on_king"] = num_attackers_on_enemy_king(board)
    feats["num_pinned_pieces"] = num_pinned_pieces(board, board.turn)

    return feats


# =========================================================
# Controller — Token Bucket Policy
# =========================================================

@dataclass
class ControllerDecision:
    """Captures everything the controller decided for one move."""
    predicted_bucket: int
    chosen_bucket: int
    remaining_budget: int
    moves_left_estimate: int
    tokens_before: float        # token level before spending
    tokens_after: float         # token level after spending (before refill)
    refill_rate: float          # current dynamic refill rate


class ModelTimeController:
    """
    Allocates a per-move node budget using a token-bucket policy.

    Token bucket mechanics:
      1. A token counter starts at `burst_cap` (the bucket is full).
      2. Each move, the classifier predicts a bucket.  The controller
         spends min(predicted_bucket, available_tokens, remaining_budget),
         snapped down to the largest affordable discrete bucket.
      3. After spending, `refill_rate` tokens are added, capped at
         `burst_cap`.
      4. `refill_rate` is recalculated every move as
             remaining_budget / estimated_moves_left
         so the sustainable pace adapts as the game progresses.

    This allows *bursting* on hard positions (spending more than the
    sustainable rate) while naturally throttling back during easy
    stretches, because tokens accumulate up to the cap.
    """

    def __init__(
        self,
        model_path: str,
        total_node_budget: int = DEFAULT_TOTAL_NODE_BUDGET,
        burst_cap: int = DEFAULT_BURST_CAP,
        expected_game_length: int = DEFAULT_EXPECTED_GAME_LENGTH,
    ):
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_cols: List[str] = payload["feature_cols"]

        # Global budget tracking
        self.total_node_budget = total_node_budget
        self.remaining_budget = total_node_budget

        # Token bucket state
        self.burst_cap = burst_cap
        self.tokens = float(burst_cap)  # start full — allows aggressive opening

        # Seed the refill rate from the initial game length estimate.
        # This gets recalculated dynamically every move.
        self.refill_rate = total_node_budget / expected_game_length

    # ---------------------------------------------------------
    # Moves-left estimation
    # ---------------------------------------------------------

    def estimate_moves_left(self, board: chess.Board) -> int:
        """
        Heuristic: piece-count proxy for remaining game length.
        Counts only non-king pieces.  Returns *per-side* moves, i.e.
        half-moves that this side still has to play.
        """
        x = sum(
            1 for p in board.piece_map().values()
            if p.piece_type != chess.KING
        )

        if x < 20:
            y = x + 10
        elif x <= 60:
            y = (3 / 8) * x + 22
        else:
            y = (5 / 4) * x - 30

        return max(1, int(y / 2))


    # ---------------------------------------------------------
    # Token bucket allocation
    # ---------------------------------------------------------

    def _snap_to_bucket(self, max_nodes: float) -> int:
        """
        Return the largest discrete bucket that fits within `max_nodes`.
        If none fits, return the smallest bucket clamped to remaining budget.
        """
        affordable = [b for b in BUCKETS if b <= max_nodes]
        if affordable:
            return max(affordable)
        # Cannot afford even the smallest bucket — spend what we can
        return min(int(max_nodes), BUCKETS[0])

    def choose_bucket_token(self, predicted_bucket: int) -> int:
        """
        Token bucket spending rule:
          - You may spend up to min(predicted, tokens, remaining_budget).
          - The result is snapped down to the nearest discrete bucket.
        """
        spendable = min(
            predicted_bucket,
            self.tokens,
            self.remaining_budget,
        )
        return self._snap_to_bucket(spendable)

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict_bucket(self, board: chess.Board) -> int:
        feats = extract_static_features(board)
        row = {col: feats.get(col, 0) for col in self.feature_cols}
        X = pd.DataFrame([row], columns=self.feature_cols)
        return int(self.model.predict(X)[0])

    # ---------------------------------------------------------
    # Main decision entry point
    # ---------------------------------------------------------

    def decide(self, board: chess.Board) -> ControllerDecision:
        # 1. Update refill rate based on current position
        moves_left = self.estimate_moves_left(board)
        self.refill_rate = self.remaining_budget / max(moves_left, 1)

        # 2. Classifier predicts the ideal bucket
        predicted_bucket = self.predict_bucket(board)

        # 3. Token bucket decides what we can actually afford
        tokens_before = self.tokens
        chosen_bucket = self.choose_bucket_token(predicted_bucket)

        return ControllerDecision(
            predicted_bucket=predicted_bucket,
            chosen_bucket=chosen_bucket,
            remaining_budget=self.remaining_budget,
            moves_left_estimate=moves_left,
            tokens_before=tokens_before,
            tokens_after=tokens_before,   # filled in after consume
            refill_rate=self.refill_rate,
        )

    # ---------------------------------------------------------
    # Post-move bookkeeping
    # ---------------------------------------------------------

    def consume_budget(self, nodes_used: int) -> None:
        """
        Called after the engine returns the actual node count used.
        Deducts from both the global budget and the token counter,
        then refills the token counter by the current refill rate.
        """
        # Deduct from global budget
        self.remaining_budget = max(0, self.remaining_budget - nodes_used)

        # Deduct from token counter
        self.tokens = max(0.0, self.tokens - nodes_used)

        # Refill — the drip that lets savings accumulate
        self.tokens = min(
            self.tokens + self.refill_rate,
            float(self.burst_cap),
        )


# =========================================================
# Play loop
# =========================================================

def play_with_controller(
    fen: Optional[str] = None,
    max_plies: int = 200,
    verbose: bool = True,
    pgn_path: str = "games/controller_game.pgn",
):
    board = chess.Board(fen) if fen else chess.Board()
    controller = ModelTimeController(
        MODEL_PATH,
        total_node_budget=DEFAULT_TOTAL_NODE_BUDGET,
    )

    game = chess.pgn.Game()
    game.headers["Event"] = "Token Bucket Controller Self-Play"
    game.headers["White"] = "Controller"
    game.headers["Black"] = "Controller"
    game.headers["Site"] = "Local"
    game.headers["Round"] = "-"
    game.headers["Result"] = "*"

    if fen:
        game.headers["FEN"] = board.fen()
        game.headers["SetUp"] = "1"

    node = game
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1, "Hash": 128})

    try:
        ply = 0

        while not board.is_game_over() and ply < max_plies:
            decision = controller.decide(board)

            info = engine.analyse(
                board,
                chess.engine.Limit(nodes=decision.chosen_bucket),
            )

            pv = info.get("pv", [])
            if not pv:
                print("No PV returned. Stopping.")
                break

            move = pv[0]
            nodes_used = info.get("nodes", decision.chosen_bucket)

            if verbose:
                print("=" * 60)
                print(f"Ply:              {ply + 1}")
                print(f"FEN:              {board.fen()}")
                print(f"Predicted bucket: {decision.predicted_bucket:>12,}")
                print(f"Chosen bucket:    {decision.chosen_bucket:>12,}")
                print(f"Nodes used:       {nodes_used:>12,}")
                print(f"Tokens before:    {decision.tokens_before:>12,.0f}")
                print(f"Refill rate:      {decision.refill_rate:>12,.0f}")
                print(f"Remaining budget: {controller.remaining_budget:>12,}")
                print(f"Moves left est:   {decision.moves_left_estimate}")
                print(f"Move played:      {move.uci()}")

            node = node.add_variation(move)
            node.comment = (
                f"predicted={decision.predicted_bucket}, "
                f"chosen={decision.chosen_bucket}, "
                f"nodes_used={nodes_used}, "
                f"tokens_before={decision.tokens_before:.0f}, "
                f"refill_rate={decision.refill_rate:.0f}, "
                f"remaining_budget={controller.remaining_budget}, "
                f"moves_left={decision.moves_left_estimate}"
            )

            board.push(move)
            controller.consume_budget(nodes_used)
            ply += 1

            if controller.remaining_budget <= 0:
                print("Budget exhausted. Stopping.")
                break

        if board.is_game_over():
            game.headers["Result"] = board.result()
        else:
            game.headers["Result"] = "*"

        print("=" * 60)
        print("Final fen:", board.fen())
        print("Final board:")
        print(board)
        print("Result:", board.result() if board.is_game_over() else "unfinished")
        print("Remaining budget:", controller.remaining_budget)
        print(f"Tokens remaining: {controller.tokens:,.0f}")

        with open(pgn_path, "w", encoding="utf-8") as f:
            print(game, file=f, end="\n\n")

        print(f"PGN saved to: {pgn_path}")

    except Exception as e:
        print(f"Error during play: {e}")
        raise

    finally:
        try:
            engine.quit()
        except Exception:
            try:
                engine.close()
            except Exception:
                pass


if __name__ == "__main__":
    play_with_controller(
        "rn1q1rk1/1pp2pp1/4pb1p/p7/P2P4/5BP1/1P1NPP1P/R1QR2K1 b - - 0 16"
    )