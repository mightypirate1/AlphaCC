"""Verify Board cached fields (hash, reward, winner) are never stale.

These were introduced in 90f0bdf and are used by the MCTS tree (DashMap keying)
and by get_info() (game_over / reward). If any cached value is wrong, MCTS
silently breaks: tree lookups miss, games terminate incorrectly, or value
targets are garbage.
"""

import pickle

import numpy as np
import pytest

from alpha_cc.engine import Board


def _fresh_reward_and_winner(board: Board) -> tuple[float, int]:
    """Recompute reward/winner from scratch via a serialize round-trip.

    After bitcode decode the skip-fields are zeroed; deserialize_rs then
    calls recompute_cache(). Comparing against the live board tells us
    whether the *original* board's cache drifted.
    """
    raw = board.__getstate__()
    fresh = Board.__new__(Board)
    fresh.__setstate__(raw)
    return fresh.info.reward, fresh.info.winner


def _fresh_hash(board: Board) -> int:
    """Get the hash of a board reconstructed from serialized bytes."""
    raw = board.__getstate__()
    fresh = Board.__new__(Board)
    fresh.__setstate__(raw)
    return hash(fresh)


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [3, 5, 7, 9])
def test_cache_valid_after_create(size: int) -> None:
    board = Board(size)
    assert hash(board) == _fresh_hash(board)
    r, w = _fresh_reward_and_winner(board)
    assert board.info.reward == pytest.approx(r)
    assert board.info.winner == w


@pytest.mark.parametrize("size", [3, 5, 7])
def test_cache_valid_through_full_game(size: int) -> None:
    """Play a full game and verify cache at every step."""
    board = Board(size)
    step = 0
    while not board.info.game_over:
        step += 1
        assert step < 200000, "game seems infinite"

        # verify hash
        assert hash(board) == _fresh_hash(board), f"hash mismatch at step {step}"

        # verify reward/winner
        fresh_r, fresh_w = _fresh_reward_and_winner(board)
        assert board.info.reward == pytest.approx(
            fresh_r
        ), f"reward mismatch at step {step}: cached={board.info.reward}, fresh={fresh_r}"
        assert (
            board.info.winner == fresh_w
        ), f"winner mismatch at step {step}: cached={board.info.winner}, fresh={fresh_w}"

        # verify game_over consistent with winner
        assert board.info.game_over == (board.info.winner > 0)

        # advance
        moves = board.get_moves()
        board = board.apply(moves[np.random.randint(len(moves))])

    # final board should also be valid
    assert hash(board) == _fresh_hash(board)
    fresh_r, fresh_w = _fresh_reward_and_winner(board)
    assert board.info.reward == pytest.approx(fresh_r)
    assert board.info.winner == fresh_w
    assert board.info.game_over


@pytest.mark.parametrize("size", [3, 5, 7])
def test_cache_survives_pickle_roundtrip(size: int) -> None:
    board = Board(size)
    for _ in range(10):
        if board.info.game_over:
            break
        moves = board.get_moves()
        board = board.apply(moves[np.random.randint(len(moves))])

    restored = pickle.loads(pickle.dumps(board))  # noqa: S301
    assert hash(restored) == hash(board)
    assert restored.info.reward == pytest.approx(board.info.reward)
    assert restored.info.winner == board.info.winner
    assert restored.info.game_over == board.info.game_over


@pytest.mark.parametrize("size", [3, 5, 7])
def test_clone_preserves_hash(size: int) -> None:
    """board.apply() produces a board; calling get_moves()+apply() on the
    same parent with the same move index must give an equal board with
    equal hash.  This is the exact pattern used in the training loop vs
    the MCTS tree."""
    board = Board(size)
    for _ in range(5):
        if board.info.game_over:
            break
        moves = board.get_moves()
        a = np.random.randint(len(moves))

        board_a = board.apply(moves[a])

        # re-fetch moves (fresh call, like training_runtime does)
        moves_again = board.get_moves()
        board_b = board.apply(moves_again[a])

        assert hash(board_a) == hash(board_b), "same parent + same action index produced different hashes"
        assert board_a == board_b
        assert board_a.info.reward == pytest.approx(board_b.info.reward)
        assert board_a.info.winner == board_b.info.winner

        board = board_a


@pytest.mark.parametrize("size", [3, 5, 7])
def test_move_ordering_deterministic(size: int) -> None:
    """get_moves() must return the same order for the same board,
    regardless of how that board was obtained."""
    board = Board(size)
    for _ in range(8):
        if board.info.game_over:
            break
        moves_1 = board.get_moves()
        moves_2 = board.get_moves()
        assert len(moves_1) == len(moves_2)
        for m1, m2 in zip(moves_1, moves_2):
            assert m1.from_coord.x == m2.from_coord.x
            assert m1.from_coord.y == m2.from_coord.y
            assert m1.to_coord.x == m2.to_coord.x
            assert m1.to_coord.y == m2.to_coord.y

        # also check: board from tree path vs board from independent apply
        a = np.random.randint(len(moves_1))
        child = board.apply(moves_1[a])
        child_independent = board.apply(board.get_moves()[a])

        child_moves = child.get_moves()
        indep_moves = child_independent.get_moves()
        assert len(child_moves) == len(indep_moves)
        for m1, m2 in zip(child_moves, indep_moves):
            assert m1.from_coord.x == m2.from_coord.x
            assert m1.from_coord.y == m2.from_coord.y
            assert m1.to_coord.x == m2.to_coord.x
            assert m1.to_coord.y == m2.to_coord.y

        board = child
