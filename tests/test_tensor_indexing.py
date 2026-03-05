import numpy as np
import pytest
import torch

from alpha_cc.engine import build_inference_request, preds_from_logits
from alpha_cc.engine.engine_utils import action_indexer
from alpha_cc.state import GameState
from alpha_cc.state.state_tensors import state_tensor

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


@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_preds_from_logits_indexing(size: int) -> None:
    """Verify that Rust preds_from_logits extracts the correct move logits
    and produces valid softmax probabilities."""
    board = get_random_board_state(size)
    state = GameState(board)
    moves = state.moves
    n_moves = len(moves)

    # Create logits where each move's 4D position has a unique, known value.
    # All non-move positions are -inf so softmax concentrates on legal moves.
    logits_4d = torch.full((size, size, size, size), float("-inf"))
    expected_logits = torch.randn(n_moves)
    for i, move in enumerate(moves):
        fx, fy = move.from_coord.x, move.from_coord.y
        tx, ty = move.to_coord.x, move.to_coord.y
        logits_4d[fx, fy, tx, ty] = expected_logits[i]

    values = np.array([0.42], dtype=np.float32)
    logits_flat = logits_4d.numpy().ravel()

    preds = preds_from_logits(logits_flat, values, [board], size)
    assert len(preds) == 1
    pred = preds[0]

    # Verify softmax was applied to the correct logits
    expected_pi = torch.softmax(expected_logits, dim=0).numpy()
    np.testing.assert_allclose(pred.pi, expected_pi, atol=1e-5)  # Q0.16: 7.6e-6 per element
    np.testing.assert_allclose(pred.value, 0.42, atol=2e-5)  # Q1.15: 1.5e-5


@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_preds_from_logits_matches_action_indexer(size: int) -> None:
    """Verify that Rust flat indexing extracts the same values as Python action_indexer."""
    board = get_random_board_state(size)
    state = GameState(board)

    logits_4d = torch.randn(size, size, size, size)
    logits_flat = logits_4d.numpy().ravel()
    values = np.array([0.0], dtype=np.float32)

    # Python path: action_indexer on the 4D tensor
    idx = action_indexer(state)
    py_move_logits = logits_4d.numpy()[*idx]
    py_pi = np.exp(py_move_logits - py_move_logits.max())
    py_pi /= py_pi.sum()

    # Rust path: preds_from_logits on the flattened array
    preds = preds_from_logits(logits_flat, values, [board], size)
    rust_pi = np.array(preds[0].pi)

    np.testing.assert_allclose(rust_pi, py_pi, atol=1e-5)


@pytest.mark.parametrize("size", [3, 5, 7, 9])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_preds_from_logits_batched(size: int, batch_size: int) -> None:
    """Verify batched preds_from_logits handles multiple boards correctly."""
    boards = [get_random_board_state(size) for _ in range(batch_size)]
    logits = torch.randn(batch_size, size, size, size, size)
    values = torch.rand(batch_size) * 2 - 1  # uniform in [-1, 1] (valid for NNQuantizedValue)

    logits_flat = logits.numpy().ravel()
    values_flat = values.numpy().ravel()

    preds = preds_from_logits(logits_flat, values_flat, boards, size)
    assert len(preds) == batch_size

    for i, (board, pred) in enumerate(zip(boards, preds)):
        state = GameState(board)
        n_moves = len(state.moves)
        assert len(pred.pi) == n_moves
        # Q0.16: 7.6e-6 per element, N moves worst case
        np.testing.assert_allclose(sum(pred.pi), 1.0, atol=n_moves * 8e-6)
        np.testing.assert_allclose(pred.value, values[i].item(), atol=2e-5)  # Q1.15: 1.5e-5


@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_build_inference_request(size: int) -> None:
    """Verify Rust build_inference_request produces correct tensor and move coords."""
    boards = [get_random_board_state(size) for _ in range(16)]

    for board in boards:
        tensor, move_coords = build_inference_request(board)

        # Tensor matches Python reference
        state = GameState(board)
        py_tensor = state_tensor(state)
        torch.testing.assert_close(torch.from_numpy(tensor), py_tensor)

        # Move coords match board.get_moves()
        moves = board.get_moves()
        assert len(move_coords) == len(moves)
        for (fx, fy, tx, ty), move in zip(move_coords, moves):
            assert (fx, fy, tx, ty) == (
                move.from_coord.x,
                move.from_coord.y,
                move.to_coord.x,
                move.to_coord.y,
            )
