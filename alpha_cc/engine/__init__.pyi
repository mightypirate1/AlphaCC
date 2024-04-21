"""
Typehints and docs for the game engine

"""

import numpy as np

class Board:
    def __init__(self, size: int) -> None:
        """Board with side length `size`"""

    def reset(self) -> Board:
        """Resets board to starting position, randomizes starting player, and returns the Board"""

    def reset_with_starting_player(self, starting_player: int) -> Board:
        """Resets board to starting position and sets current player, and returns the Board"""

    def get_legal_moves(self) -> Moves:
        """Get all moves allowed in the current position"""

    def get_all_possible_next_states(self) -> list[Board]:
        """Returns a list of all board configurations that can be reached through a legal move"""

    def perform_move(self, move_index: int | np.integer) -> Board:
        """
        Performs the move with index `move_index`.

        Returns a copy, so the original board is unchanged.

        The index refers to the list returned by `get_all_possible_next_states`
        """

    def get_matrix(self) -> list[list[int]]:
        """Raw data from the board"""

    def get_matrix_from_perspective_of_player(self, player: int) -> list[list[int]]:
        """Raw data from the board as seen from `player`'s perspective"""

    def get_matrix_from_perspective_of_current_player(self) -> list[list[int]]:
        """Raw data from the board as seen from the current player's perspective"""

    def render(self) -> None:
        """Print the matrix"""

    @property
    def size(self) -> int:
        """Size of the board. Actual matrix shape is (size, size)"""

    @property
    def board_info(self) -> BoardInfo:
        """Get the `BoardInfo`"""

class BoardInfo:
    current_player: int
    game_over: bool
    winner: int
    size: int
    duration: int

    def __init__(self, size: int) -> None: ...

class HexCoordinate:
    x: int
    y: int

class _Move:
    from_coord: HexCoordinate
    to_coord: HexCoordinate

class Move(_Move):  # TODO: work on better typing here (may need pyo3 changes)
    """
    `Move` is a rust enum with types `Jump` and `Walk`.

    NOTE: type hints are not perfect for `Move`!

    """

    class Walk(_Move): ...
    class Jump(_Move): ...

class Moves(tuple[Move]):
    def get_action_mask(self) -> list[list[list[list[int]]]]:
        """
        Given a `board_size`, create a mask that shows what moves are allowed
        according to the following convention:
        ```
        is_allowed = mask[from_x][from_y][to_x][to_y]
        ```
        """

    def get_action_mask_indices(self) -> dict[int, tuple[HexCoordinate, HexCoordinate]]:
        """
        Returns a mapping that can help you find moves in the mask from `Moves.to_mask`.

        E.g.
        moves = board.get_legal_moves()
        move_mask = moves.to_mask()
        action_mask_indices = moves.get_action_mask_indices()

        for i, move in enumerate(moves):
            from_coord, to_coord = action_mask_indices[i]
            assert move_mask[from_coord.x][from_coord.y][to_coord.x][to_coord.y] == i

        """
