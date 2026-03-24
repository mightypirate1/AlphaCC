from collections import Counter

import numpy as np

from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.engine import Board
from alpha_cc.state.game_state import GameState
from alpha_cc.training import TrainingDataset


def _make_experience(state: GameState, v_target: float = 0.0) -> MCTSExperience:
    n = len(state.children)
    return MCTSExperience(
        state,
        pi_target=np.full(n, 1.0 / n),
        v_target=v_target,
    )


def test_ring_buffer_basic() -> None:
    state = GameState(Board(5))
    dataset = TrainingDataset(max_size=10)
    batch = [_make_experience(state, v_target=float(i)) for i in range(5)]
    dataset.add_trajectory(batch)
    assert len(dataset) == 5
    assert all(exp is not None for exp in dataset.samples)


def test_ring_buffer_wraps() -> None:
    state = GameState(Board(5))
    dataset = TrainingDataset(max_size=5)
    # Add 5 samples
    batch_a = [_make_experience(state, v_target=1.0) for _ in range(5)]
    dataset.add_trajectory(batch_a)
    assert len(dataset) == 5

    # Add 3 more — oldest 3 should be evicted
    batch_b = [_make_experience(state, v_target=-1.0) for _ in range(3)]
    dataset.add_trajectory(batch_b)
    assert len(dataset) == 5
    counter = Counter(exp.v_target for exp in dataset.samples)
    assert counter[1.0] == 2
    assert counter[-1.0] == 3


def test_prioritized_sample_returns_all_when_n_exceeds_size() -> None:
    state = GameState(Board(5))
    dataset = TrainingDataset(max_size=10)
    batch = [_make_experience(state) for _ in range(5)]
    dataset.add_trajectory(batch)
    indices, sampled = dataset.prioritized_sample(20)
    assert len(indices) == 5
    assert len(sampled) == 5


def test_prioritized_sample_biases_toward_high_error() -> None:
    """After updating priorities, high-error samples should be sampled more often."""
    state = GameState(Board(5))
    dataset = TrainingDataset(max_size=100, gamma=0.5)

    # Add 100 samples
    batch = [_make_experience(state, v_target=float(i)) for i in range(100)]
    dataset.add_trajectory(batch)

    # Set priorities: first 10 samples have very high error, rest low
    high_indices = np.arange(10)
    low_indices = np.arange(10, 100)
    dataset.update_priorities(
        high_indices,
        kl_divs=np.full(10, 10.0, dtype=np.float32),
        td_errors=np.full(10, 10.0, dtype=np.float32),
    )
    dataset.update_priorities(
        low_indices,
        kl_divs=np.full(90, 0.01, dtype=np.float32),
        td_errors=np.full(90, 0.01, dtype=np.float32),
    )

    # Sample 20 items many times and check high-error samples are overrepresented
    high_count = 0
    n_trials = 200
    for _ in range(n_trials):
        indices, _ = dataset.prioritized_sample(20)
        high_count += np.isin(indices, high_indices).sum()

    # With uniform sampling, we'd expect 10/100 * 20 = 2 high-error per sample = 400 total
    # With PER, high-error should be sampled much more often
    expected_uniform = n_trials * 20 * (10 / 100)
    assert high_count > expected_uniform * 2, f"PER not biasing enough: {high_count} vs uniform {expected_uniform}"


def test_update_priorities() -> None:
    state = GameState(Board(5))
    dataset = TrainingDataset(max_size=10)
    batch = [_make_experience(state) for _ in range(5)]
    dataset.add_trajectory(batch)

    indices = np.array([0, 2, 4])
    kl_divs = np.array([5.0, 3.0, 1.0], dtype=np.float32)
    td_errors = np.array([0.5, 0.3, 0.1], dtype=np.float32)
    dataset.update_priorities(indices, kl_divs, td_errors)

    # Verify values were written
    assert dataset._kl_div[0] == 5.0
    assert dataset._kl_div[2] == 3.0
    assert abs(dataset._td_error[4] - 0.1) < 1e-6


def test_split() -> None:
    state = GameState(Board(5))
    dataset = TrainingDataset(max_size=100)
    batch = [_make_experience(state) for _ in range(100)]
    dataset.add_trajectory(batch)
    train, test = dataset.split(0.8)
    assert len(train) == 80
    assert len(test) == 20
