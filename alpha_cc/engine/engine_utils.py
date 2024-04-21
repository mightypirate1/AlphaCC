import numpy as np

from alpha_cc.engine import Board


def action_indexer(board: Board) -> np.ndarray:
    move_mask_indices = board.get_legal_moves().get_action_mask_indices()
    return np.array(
        [
            [
                move_mask_indices[i][0].x,
                move_mask_indices[i][0].y,
                move_mask_indices[i][1].x,
                move_mask_indices[i][1].y,
            ]
            for i in range(len(board.get_all_possible_next_states()))
        ]
    ).T
