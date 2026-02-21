"""
Typehints and docs for the game engine

"""

import numpy

def create_move_mask(moves: list[Move]) -> list[list[list[list[bool]]]]: ...
def create_move_index_map(moves: list[Move]) -> dict[int, tuple[HexCoord, HexCoord]]: ...
def boards_to_state_tensor(
    boards: list[Board],
    board_size: int,
) -> numpy.ndarray:
    """Convert boards to state tensor with shape (batch, 2, size, size) as float32."""
    ...

def preds_from_logits(
    logits_flat: numpy.ndarray,
    values_flat: numpy.ndarray,
    boards: list[Board],
    board_size: int,
) -> list[NNPred]:
    """Compute NNPreds from raw logits (softmax over legal moves).

    Arrays are read zero-copy from numpy buffers. Both must be contiguous float32.
    """
    ...

def post_preds_from_logits(
    pred_db: PredDBChannel,
    logits_flat: numpy.ndarray,
    values_flat: numpy.ndarray,
    boards: list[Board],
    board_size: int,
) -> None:
    """Like preds_from_logits, but writes results directly to memcached."""
    ...

class Board:
    def __init__(self, size: int) -> None:
        """Board with side length `size`"""

    def reset(self) -> Board: ...
    def get_moves(self) -> list[Move]:
        """Get all moves allowed in the current position"""

    def get_next_states(self) -> list[Board]:
        """Returns a list of all board configurations that can be reached through a legal move"""

    def apply(self, move: Move) -> Board:
        """
        Performs the move with index `move_index`.

        Returns a copy, so the original board is unchanged.

        The index refers to the list returned by `get_next_states`
        """

    def get_matrix(self) -> list[list[int]]:
        """Raw data from the board"""

    def get_unflipped_matrix(self) -> list[list[int]]:
        """Raw data from the board as seen from `player`'s perspective"""

    def render(self) -> None:
        """Print the matrix"""

    @property
    def info(self) -> BoardInfo:
        """Get the `BoardInfo`"""

    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

class BoardInfo:
    current_player: int
    game_over: bool
    winner: int
    size: int
    reward: float
    duration: int

    def __init__(self, size: int) -> None: ...

class FetchStats:
    """Per-game fetch statistics from the MCTS worker's NN prediction loop."""

    @property
    def resolved_at_attempt(self) -> list[int]:
        """Count of fetches resolved at each attempt (0 = cache hit, 1 = first patience, 2+ = backoff)."""
        ...

    @property
    def attempt_total_wait_us(self) -> list[int]:
        """Sum of elapsed time (microseconds) for fetches resolved at each attempt."""
        ...

    @property
    def timeouts(self) -> int: ...
    @property
    def total_gets(self) -> int: ...
    @property
    def total_misses(self) -> int: ...
    @property
    def total_fetch_time_us(self) -> int: ...
    @property
    def total_fetches(self) -> int: ...
    @property
    def current_patience_us(self) -> int: ...

class HexCoord:
    x: int
    y: int

    def get_all_neighbors(self, distance: int) -> list[HexCoord]: ...
    def flip(self) -> HexCoord: ...

class Move:
    from_coord: HexCoord
    to_coord: HexCoord

class MCTSNode:
    @property
    def n(self) -> list[int]: ...
    @property
    def q(self) -> list[float]: ...
    @property
    def pi(self) -> list[float]: ...
    @property
    def v(self) -> float: ...
    @property
    def moves(self) -> list[Move]: ...

class MCTS:
    def __init__(
        self,
        keydb_url: str,
        memcached_url: str,
        channel: int,
        cache_size: int,
        rollout_gamma: float,
        dirichlet_weight: float,
        dirichlet_alpha: float,
        c_puct_init: float,
        c_puct_base: float,
    ) -> None: ...
    def clear_nodes(self) -> None: ...
    def get_node(self, board: Board) -> MCTSNode | None: ...
    def get_nodes(self) -> dict[Board, MCTSNode]: ...
    def run(self, board: Board, rollout_depth: int) -> float: ...
    def get_fetch_stats(self) -> FetchStats:
        """Read and reset fetch statistics. Returns a snapshot of all counters since last call."""
        ...

class NNPred:
    @property
    def pi(self) -> list[float]: ...
    @property
    def value(self) -> float: ...
    def __init__(self, pi: list[float], value: float) -> None: ...

class PredDBChannel:
    def __init__(self, keydb_url: str, memcached_url: str, channel: int) -> None: ...
    @property
    def channel(self) -> int: ...
    def ping(self) -> bool:
        """Tests if channel is connected correctly"""

    def get_channel(self) -> int:
        """Get the channel number"""

    def has_pred(self, board: Board) -> bool:
        """Check if a prediction is available for the given board"""

    def fetch_requests(self, count: int) -> list[Board]:
        """Pop pred requests from the channel"""

    def request_pred(self, board: Board) -> None:
        """Request a prediction for the given board"""

    def post_pred(self, board: Board, nn_pred: NNPred) -> None:
        """Post a prediction for the given board"""

    def post_preds(self, boards: list[Board], nn_preds: list[NNPred]) -> None:
        """Post a list of predictions for the given boards"""

    def flush_preds(self) -> None: ...
