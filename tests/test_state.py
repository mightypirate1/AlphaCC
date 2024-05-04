import dill
import pytest

from alpha_cc.engine import Board
from alpha_cc.nn.nets import DefaultNet
from alpha_cc.state import GameState


@pytest.mark.parametrize("board_size", [3, 5, 7, 9])
def test_properties(board_size: int) -> None:
    state = GameState(Board(board_size))
    # unclear what to test here, but we can check that it at least doesnt crash
    state.board  # noqa
    state.info  # noqa
    state.moves  # noqa
    state.children  # noqa
    state.action_mask  # noqa
    state.action_mask_indices  # noqa
    state.hash  # noqa
    state.matrix  # noqa
    state.tensor  # noqa


@pytest.mark.parametrize("board_size", [3, 5, 7, 9])
def test_representations(board_size: int) -> None:
    state = GameState(Board(board_size))
    assert state.matrix.shape == (board_size, board_size)
    assert state.tensor.shape == (2, board_size, board_size)


@pytest.mark.parametrize("board_size", [3, 5, 7, 9])
def test_pickling(board_size: int) -> None:
    state = GameState(Board(board_size))
    state_bytes = dill.dumps(state)
    recreated_state = dill.loads(state_bytes)  # noqa
    # all fields recreated
    state.board  # noqa
    state.info  # noqa
    state.moves  # noqa
    state.children  # noqa
    state.action_mask  # noqa
    state.action_mask_indices  # noqa
    state.hash  # noqa
    state.matrix  # noqa
    state.tensor  # noqa
    assert (recreated_state.matrix == state.matrix).all()


@pytest.mark.parametrize("board_size", [3, 5, 7, 9])
def test_nn(board_size: int) -> None:
    state = GameState(Board(board_size))
    nn = DefaultNet(board_size)
    pi, v = nn(state.tensor.unsqueeze(0))
    assert pi.shape == (1, board_size, board_size, board_size, board_size)
    assert v.shape == (1,)
