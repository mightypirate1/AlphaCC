import dill
import pytest

from alpha_cc.engine import Board, HexCoord
from alpha_cc.nn.nets import DefaultNet
from alpha_cc.state import GameState

from .common import get_random_board_state


@pytest.mark.parametrize("board_size", [3, 5, 7, 9])
def test_properties(board_size: int) -> None:
    state = GameState(get_random_board_state(board_size))
    # unclear what to test here, but we can check that it at least doesnt crash
    state.board  # noqa
    state.info  # noqa
    state.moves  # noqa
    state.children  # noqa
    state.action_mask  # noqa
    state.action_mask_indices  # noqa
    state.hash  # noqa
    state.matrix  # noqa
    state.unflipped_matrix  # noqa
    state.tensor  # noqa


@pytest.mark.parametrize("board_size", [3, 5, 7, 9])
def test_representations(board_size: int) -> None:
    state = GameState(get_random_board_state(board_size))
    assert state.matrix.shape == (board_size, board_size)
    assert state.tensor.shape == (2, board_size, board_size)
    assert state.action_mask.sum() == len(state.action_mask_indices)


@pytest.mark.parametrize("board_size", [3, 5, 7, 9])
def test_pickling(board_size: int) -> None:
    def assert_coords_eq(coord_a: HexCoord, coord_b: HexCoord) -> None:
        assert coord_a.x == coord_b.x
        assert coord_a.y == coord_b.y

    state = GameState(get_random_board_state(board_size))
    state_bytes = dill.dumps(state)
    recreated_state = dill.loads(state_bytes)  # noqa
    # all fields recreated?
    recreated_state.board  # noqa
    recreated_state.info  # noqa
    assert state.info.current_player == recreated_state.info.current_player
    assert state.info.duration == recreated_state.info.duration
    assert (state.action_mask == recreated_state.action_mask).all()
    assert (state.tensor == recreated_state.tensor).all()
    assert len(state.action_mask_indices) == len(recreated_state.action_mask_indices)
    for i, (from_coord, to_coord) in state.action_mask_indices.items():
        rec_from_coord, rec_to_coord = recreated_state.action_mask_indices[i]
        assert_coords_eq(from_coord, rec_from_coord)
        assert_coords_eq(to_coord, rec_to_coord)

    assert len(state.children) == len(recreated_state.children)
    assert (state.tensor == recreated_state.tensor).all()
    assert (recreated_state.matrix == state.matrix).all()
    assert (recreated_state.unflipped_matrix == state.unflipped_matrix).all()
    assert state.hash == recreated_state.hash
    for move, rec_move in zip(state.moves, recreated_state.moves, strict=True):
        assert_coords_eq(move.from_coord, rec_move.from_coord)
        assert_coords_eq(move.to_coord, rec_move.to_coord)


@pytest.mark.parametrize("board_size", [3, 5, 7, 9])
def test_nn(board_size: int) -> None:
    state = GameState(Board(board_size))
    nn = DefaultNet(board_size)
    pi, v = nn(state.tensor.unsqueeze(0))
    assert pi.shape == (1, board_size, board_size, board_size, board_size)
    assert v.shape == (1,)
