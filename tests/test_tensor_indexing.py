import numpy as np
import pytest

from alpha_cc.engine.engine_utils import action_indexer
from alpha_cc.state import GameState

from .common import get_random_board_state


@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_move_indexer(size: int) -> None:
    board = get_random_board_state(size)
    state = GameState(board)

    batch_size = 32
    numel = batch_size * size**4
    tensor_like = np.arange(numel).reshape((batch_size, size, size, size, size))
    vectorized = tensor_like[:, *action_indexer(state)]
    assert len(state.moves) == vectorized.shape[1]

    for i in range(batch_size):
        for j, move in enumerate(state.moves):
            fx = move.from_coord.x
            fy = move.from_coord.y
            tx = move.to_coord.x
            ty = move.to_coord.y
            assert vectorized[i, j] == tensor_like[i][fx][fy][tx][ty]
