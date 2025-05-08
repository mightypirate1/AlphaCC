from collections import Counter

import numpy as np

from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.engine import Board
from alpha_cc.state.game_state import GameState
from alpha_cc.training import TrainingDataset


def test_correct_sampling() -> None:
    dataset = TrainingDataset(max_size=10)
    state = GameState(Board(5))
    n = len(state.children)
    batch_a_value = 1.0
    batch_b_value = -1.0
    experience_batch_a = [
        MCTSExperience(
            state,
            pi_target=np.full(n, 1.0 / n),
            v_target=batch_a_value,
        )
        for _ in range(5)
    ]
    experience_batch_b = [
        MCTSExperience(
            state,
            pi_target=np.full(n, 1.0 / n),
            v_target=batch_b_value,
        )
        for _ in range(5)
    ]
    # add one batch and sample it
    dataset.add_trajectory(experience_batch_a)
    sample_a = dataset.sample(5)
    assert len(dataset) == 5
    assert len(sample_a) == 5

    # add second batch and make sure we get it when we sample again
    dataset.move_new_to_main_buffer()
    dataset.add_trajectory(experience_batch_b)
    sample_b = dataset.sample(5)
    assert len(dataset) == 10
    assert len(sample_b) == 5
    assert all(exp.v_target == batch_b_value for exp in sample_b.samples)

    # make sure we didn't lose any samples
    sample_c = dataset.sample(10, replace=False)
    assert len(sample_c) == 10
    counter = Counter(exp.v_target for exp in sample_c.samples)
    assert counter[batch_a_value] == 5
    assert counter[batch_b_value] == 5
