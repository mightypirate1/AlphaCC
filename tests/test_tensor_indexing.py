import numpy as np
import pytest

from alpha_cc.engine.engine_utils import action_indexer

from .common import get_random_board_state


@pytest.mark.parametrize("board_size", range(3, 13))
def test_move_indexer(board_size: int) -> None:
    board = get_random_board_state(board_size)
    moves = board.get_legal_moves()

    batch_size = 7
    tensor_like = np.arange(batch_size * board_size**4).reshape(
        (batch_size, board_size, board_size, board_size, board_size)
    )
    vectorized = tensor_like[:, *action_indexer(board)]
    assert len(moves) == vectorized.shape[1]

    for i in range(batch_size):
        for j, move in enumerate(moves):
            fx = move.from_coord.x
            fy = move.from_coord.y
            tx = move.to_coord.x
            ty = move.to_coord.y
            assert vectorized[i, j] == tensor_like[i][fx][fy][tx][ty]
