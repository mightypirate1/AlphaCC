import numpy as np
import pytest
from alpha_cc_engine import Board


@pytest.fixture
def board() -> Board:
    return Board(9)


def test_create(board: Board) -> None:  # noqa
    pass


def test_get_data(board: Board) -> None:
    board.get_matrix()
    board.get_board_info()


def test_get_and_perform_action(board: Board) -> None:
    for _ in range(100):
        next_states = board.get_all_possible_next_states()
        action = np.random.choice(len(next_states))
        board = board.perform_move(action)


def test_rendering(board: Board) -> None:
    board.render()
