import numpy as np

from alpha_cc.engine import Board


def get_random_board_state(board_size: int) -> Board:
    board = Board(board_size)
    for _ in range(np.random.randint(10, 20)):
        if not board.info.game_over:
            moves = board.get_moves()
            a = np.random.choice(len(moves))
            move = moves[a]
            board = board.apply(move)
    return board
