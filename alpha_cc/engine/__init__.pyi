"""
Typehints and docs for the game engine

"""

import numpy

def create_move_mask(moves: list[Move]) -> list[list[list[list[bool]]]]: ...
def create_move_index_map(moves: list[Move]) -> dict[int, tuple[HexCoord, HexCoord]]: ...
def build_inference_request(
    board: Board,
) -> tuple[numpy.ndarray, list[tuple[int, int, int, int]]]:
    """Build an InferenceRequest from a Board and return its fields.

    Returns (tensor_data as ndarray (2, size, size), move_coords).
    """
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

def fetch_and_build_tensor(
    channel: int,
    max_count: int,
    board_size: int,
) -> tuple[InferenceBatch, numpy.ndarray] | None:
    """Fetch inference requests from ZMQ ROUTER, build batch tensor.

    Drains the response queue first (sends pending responses to workers),
    then receives new requests.

    Returns None if no requests available. Otherwise returns
    (InferenceBatch, ndarray with shape (batch, 2, size, size)).
    """
    ...

def enqueue_responses(
    logits_flat: numpy.ndarray,
    values_flat: numpy.ndarray,
    batch: InferenceBatch,
) -> None:
    """Enqueue inference responses for sending back to workers.

    Computes softmax over legal moves and pushes responses to the response queue.
    Responses are drained and sent by the prefetch thread via fetch_and_build_tensor.
    """
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
    def total_fetch_time_us(self) -> int: ...
    @property
    def total_fetches(self) -> int: ...

class HexCoord:
    x: int
    y: int

    def get_all_neighbors(self, distance: int) -> list[HexCoord]: ...
    def flip(self) -> HexCoord: ...

class InferenceBatch:
    """Opaque batch handle holding identities and move coords for routing responses."""

    def __len__(self) -> int: ...
    def slice(self, start: int, end: int) -> InferenceBatch:
        """Return a sub-batch from start to end."""
        ...

    def extend(self, other: InferenceBatch) -> None:
        """Append another batch's data into this one."""
        ...

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
    def run_rollouts(
        self,
        board: Board,
        n_rollouts: int,
        rollout_depth: int,
        temperature: float = 1.0,
    ) -> tuple[numpy.ndarray, float]:
        """Run n_rollouts rollouts and return (pi, mean_value).

        pi is the temperature-weighted visit count distribution.
        """
        ...

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
    def __init__(self, zmq_url: str, channel: int) -> None: ...
    @property
    def channel(self) -> int: ...
