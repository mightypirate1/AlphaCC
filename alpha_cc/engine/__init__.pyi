"""
Typehints and docs for the game engine

"""

from typing import Self

def create_move_mask(moves: list[Move]) -> list[list[list[list[bool]]]]: ...
def create_move_index_map(moves: list[Move]) -> dict[int, tuple[HexCoord, HexCoord]]: ...

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

class BoardInfo:
    current_player: int
    game_over: bool
    winner: int
    size: int
    reward: int
    duration: int

    def __init__(self, size: int) -> None: ...

class HexCoord:
    x: int
    y: int

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

class MCTS:
    def __init__(self, url: str, cache_size: int, dirichlet_weight: float, dirichlet_alpha: float) -> None:
        """
        MCTS agent

        Args:
            url: url to the model (e.g. "redis://localhost/2" for db 2 on localhost)
            cache_size: size of the MCTS search tree. If more nodes are visited,
                nodes are removed according to a LRU policy
            dirichlet_weight: weight of the dirichlet noise
            dirichlet_alpha: alpha parameter of the dirichlet noise
        """

    def get_node(self, board: Board) -> MCTSNode: ...
    def run(self, board: Board, rollout_depth: int) -> float: ...

class NNPred:
    pi: list[float]
    value: float

    def __init__(self, pi: list[float], value: float) -> None: ...
    def serialize_rs(self) -> bytes: ...
    @classmethod
    def deserialize_rs(cls: type[Self], data: bytes) -> Self: ...

class PredDB:
    def __init__(self, url: str) -> None: ...
    def fetch_all(self) -> list[Board]: ...
    def post_pred(self, board: Board, nn_pred: NNPred) -> None: ...
    def flush_preds(self) -> None: ...
