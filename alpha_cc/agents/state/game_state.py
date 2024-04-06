from alpha_cc.engine import Board, BoardInfo


class GameState:
    info: BoardInfo
    matrix: list[list[int]]

    def __init__(self, board: Board) -> None:
        self.matrix = board.get_matrix_from_perspective_of_current_player()
        self.info = board.get_board_info()
