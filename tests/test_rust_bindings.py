import numpy as np
import pytest
import torch

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


@pytest.mark.parametrize("size", [3, 5, 7, 9])
@pytest.mark.parametrize("batch_size", [1, 32, 512])
def test_post_preds_array_contiguity(size: int, batch_size: int) -> None:
    """Verify that the numpy arrays passed to post_preds_from_logits are C-contiguous,
    as required by PyO3's PyReadonlyArray1::as_slice()."""
    # Simulate NN output shapes
    x_pis = torch.randn(batch_size, size, size, size, size)
    x_vals = torch.randn(batch_size)

    # Same transforms as _post_predictions
    logits_np = x_pis.numpy().ravel()
    values_np = x_vals.numpy().ravel()

    assert logits_np.flags["C_CONTIGUOUS"], "logits array is not contiguous"
    assert values_np.flags["C_CONTIGUOUS"], "values array is not contiguous"
    assert logits_np.dtype == np.float32
    assert values_np.dtype == np.float32
