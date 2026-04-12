import numpy as np
import pytest
import torch

from alpha_cc.engine import build_inference_request, preds_from_logits
from alpha_cc.state import GameState

from .common import get_random_board_state


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

    wdl_logits = np.array([1.0, 0.0, -1.0], dtype=np.float32)  # WDL logits for 1 sample
    logits_flat = logits_4d.numpy().ravel()

    preds = preds_from_logits(logits_flat, wdl_logits, [board], size)
    assert len(preds) == 1
    pred = preds[0]

    # Verify softmax was applied to the correct logits
    expected_pi = torch.softmax(expected_logits, dim=0).numpy()
    np.testing.assert_allclose(pred.pi, expected_pi, atol=1e-5)  # Q0.16: 7.6e-6 per element
    # Verify WDL probabilities (softmax of [1, 0, -1])
    expected_wdl = torch.softmax(torch.tensor([1.0, 0.0, -1.0]), dim=0).numpy()
    np.testing.assert_allclose(pred.wdl, expected_wdl, atol=1e-5)


@pytest.mark.parametrize("size", [3, 5, 7, 9])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_preds_from_logits_batched(size: int, batch_size: int) -> None:
    """Verify batched preds_from_logits handles multiple boards correctly."""
    boards = [get_random_board_state(size) for _ in range(batch_size)]
    logits = torch.randn(batch_size, size, size, size, size)
    wdl_logits = torch.randn(batch_size, 3)  # random WDL logits

    logits_flat = logits.numpy().ravel()
    wdl_logits_flat = wdl_logits.numpy().ravel()

    preds = preds_from_logits(logits_flat, wdl_logits_flat, boards, size)
    assert len(preds) == batch_size

    expected_wdl = torch.softmax(wdl_logits, dim=-1).numpy()
    for i, (board, pred) in enumerate(zip(boards, preds)):
        state = GameState(board)
        n_moves = len(state.moves)
        assert len(pred.pi) == n_moves
        # Q0.16: 7.6e-6 per element, N moves worst case
        np.testing.assert_allclose(sum(pred.pi), 1.0, atol=n_moves * 8e-6)
        np.testing.assert_allclose(pred.wdl, expected_wdl[i], atol=1e-5)


@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_build_inference_request(size: int) -> None:
    """Verify Rust build_inference_request produces correct tensor and move coords."""
    boards = [get_random_board_state(size) for _ in range(16)]

    for board in boards:
        tensor, move_coords = build_inference_request(board)

        # Tensor should match board.state_tensor() from Rust
        expected = board.state_tensor().reshape(2, size, size)
        np.testing.assert_array_equal(tensor, expected)

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
