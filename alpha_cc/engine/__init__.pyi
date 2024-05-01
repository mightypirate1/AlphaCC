"""
Typehints and docs for the game engine

"""

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
