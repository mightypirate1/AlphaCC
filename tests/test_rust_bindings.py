import numpy as np
import pytest

from alpha_cc.engine import Board


@pytest.fixture
def board() -> Board:
    return Board(9)


def test_create(board: Board) -> None:
    pass


def test_get_data(board: Board) -> None:
    board.get_matrix()
    _ = board.info


def test_get_and_perform_action(board: Board) -> None:
    for _ in range(100):
        moves = board.get_moves()
        action = np.random.choice(len(moves))
        move = moves[action]
        board = board.apply(move)


def test_rendering(board: Board) -> None:
    board.render()
