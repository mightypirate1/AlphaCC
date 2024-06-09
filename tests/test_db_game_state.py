import numpy as np
import pytest

from alpha_cc.db.models import DBGameState
from alpha_cc.engine import Board, Move


@pytest.mark.parametrize("size", [5, 7, 9])
def test_db_game_state(size: int) -> None:
    board = Board(size)
    boards = [board]
    moves: list[Move] = []
    db_state = DBGameState("test-game", size, [])
    for _ in range(100):
        moves = board.get_moves()
        action = np.random.choice(len(moves))
        move = moves[action]
        board = board.apply(move)
        boards.append(board)
        moves.append(move)
        db_state.add_move(action)

    for board, db_state_board in zip(boards, db_state.boards, strict=True):
        assert np.array_equal(board.get_matrix(), db_state_board.get_matrix())
