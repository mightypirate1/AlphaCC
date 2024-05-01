import pytest

from alpha_cc.engine import create_move_index_map, create_move_mask

from .common import get_random_board_state


@pytest.mark.parametrize("board_size", [3, 5, 7, 9])
def test_moves(board_size: int) -> None:
    board = get_random_board_state(board_size)
    moves = board.get_moves()
    move_mask = create_move_mask(moves)
    action_mask_indices = create_move_index_map(moves)

    masked_moves = set()
    for i, _ in enumerate(moves):
        from_coord, to_coord = action_mask_indices[i]
        assert move_mask[from_coord.x][from_coord.y][to_coord.x][to_coord.y]
        masked_moves.add((from_coord.x, from_coord.y, to_coord.x, to_coord.y))

    for i in range(board_size):
        for j in range(board_size):
            for k in range(board_size):
                for l in range(board_size):  # noqa
                    if (i, j, k, l) not in masked_moves:
                        assert not move_mask[i][j][k][l]
