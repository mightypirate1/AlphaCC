"""
Typehints and docs for the game engine

"""

class Board:
    def __init__(self, size: int) -> None: ...
    def reset(self) -> Board:
        """Resets board to starting position, randomizes starting player, and returns state"""

    def get_all_possible_next_states(self) -> list[Board]:
        """Returns a list of all board configurations that can be reached through a legal move"""

    def perform_move(self, move_index: int) -> Board:
        """
        Performs the move with index `move_index`.

        Returns a copy, so the original board is unchanged.

        The index refers to the list returned by `get_all_possible_next_states`
        """

    def render(self) -> None:
        """Print the matrix"""

    def get_matrix(self) -> list[list[int]]:
        """Raw data from the board"""

    def get_matrix_from_perspective_of_player(self, player: int) -> list[list[int]]:
        """Raw data from the board as seen from `player`'s perspective"""

    def get_matrix_from_perspective_of_current_player(self) -> list[list[int]]:
        """Raw data from the board as seen from the current player's perspective"""

    def get_board_info(self) -> BoardInfo:
        """Get the `BoardInfo`"""

class BoardInfo:
    current_player: int
    game_over: bool
    winner: int

    def __init__(self, size: int) -> None: ...
