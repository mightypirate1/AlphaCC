import hashlib

from alpha_cc.engine import Board, BoardInfo


class GameState:
    info: BoardInfo
    matrix: list[list[int]]

    def __init__(self, board: Board) -> None:
        self.matrix = board.get_matrix_from_perspective_of_current_player()
        self.info = board.board_info

    def hash(self) -> bytes:
        return hashlib.sha256(
            "".join(
                [
                    str(self.matrix),
                    str(self.info.current_player),
                    str(self.info.game_over),
                    str(self.info.winner),
                ]
            ).encode()
        ).digest()
