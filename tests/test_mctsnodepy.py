import numpy as np
import pytest

from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy


@pytest.fixture
def mcts_node_py() -> MCTSNodePy:
    return MCTSNodePy(
        pi=np.array([0.2, 0.5, 0.3]),
        v_hat=0.4,
        n=np.array([2, 3, 1]),
        q=np.array([0.5, -0.6, 0.7]),
    )


def test_v(mcts_node_py: MCTSNodePy) -> None:
    assert mcts_node_py.v == 0.7


def test_flip_value(mcts_node_py: MCTSNodePy) -> None:
    flipped = mcts_node_py.with_flipped_value()
    assert flipped.v_hat == -0.4
    assert (flipped.q == np.array([-0.5, 0.6, -0.7])).all()


def test_as_sorted(mcts_node_py: MCTSNodePy) -> None:
    sorted_node = mcts_node_py.as_sorted()
    assert (sorted_node.pi == np.array([0.5, 0.2, 0.3])).all()
    assert (sorted_node.n == np.array([3, 2, 1])).all()
    assert (sorted_node.q == np.array([-0.6, 0.5, 0.7])).all()
