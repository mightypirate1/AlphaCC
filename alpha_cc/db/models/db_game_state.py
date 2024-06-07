from alpha_cc.engine import Board, Move
from alpha_cc.state import GameState


class DBGameState:
    def __init__(self, game_id: str, size: int, move_indices: list[int]) -> None:
        self._game_id = game_id
        self._size = size
        self._move_indices = move_indices
        self._moves: list[Move] = []
        self._boards = [Board(size)]

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def size(self) -> int:
        return self._size

    @property
    def move_indices(self) -> list[int]:
        return self._move_indices

    @property
    def boards(self) -> list[Board]:
        self._resolve_until_index(len(self))
        return self._boards

    @property
    def moves(self) -> list[Move]:
        self._resolve_until_index(len(self))
        return self._moves

    @property
    def states(self) -> list[GameState]:
        return [GameState(board) for board in self.boards]

    @property
    def current_game_state(self) -> GameState:
        return GameState(self.boards[-1])

    def __len__(self) -> int:
        return len(self._move_indices) + 1

    def add_move(self, move_index: int) -> None:
        self._move_indices.append(move_index)

    def get_state(self, index: int) -> GameState:
        if index >= len(self):
            raise IndexError(f"index={index} is out of bounds")
        self._resolve_until_index(index)
        board = self._boards[index]
        return GameState(board)

    def _resolve_until_index(self, index: int) -> None:
        if len(self._boards) <= index:
            board = self._boards[-1]
            for move_index in self._move_indices[len(self._boards) - 1 : index]:
                move = board.get_moves()[move_index]
                board = board.apply(move)
                self._moves.append(move)
                self._boards.append(board)
