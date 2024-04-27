import numpy as np

from alpha_cc.engine import Board


def get_random_board_state(board_size: int) -> Board:
    board = Board(board_size)
    for _ in range(np.random.randint(10, 20)):
        if not board.board_info.game_over:
            moves = board.get_legal_moves()
            board = board.perform_move(np.random.choice(len(moves)))
    return board
