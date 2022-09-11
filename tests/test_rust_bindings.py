import numpy as np
import pytest
import alpha_cc

@pytest.fixture
def board():
    return alpha_cc.Board(9)

def test_create(board):
    pass

def test_get_data(board):
    board.get_matrix()
    board.get_board_info()

def test_get_and_perform_action(board):
    for _ in range(100):
        next_states = board.get_all_possible_next_states()
        action = np.random.choice(len(next_states))
        board = board.perform_move(action)

def test_rendering(board):
    board.render()
