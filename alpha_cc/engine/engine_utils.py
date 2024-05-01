import numpy as np

from alpha_cc.state import GameState


def action_indexer(state: GameState) -> np.ndarray:
    return np.array(
        [
            [
                move.from_coord.x,
                move.from_coord.y,
                move.to_coord.x,
                move.to_coord.y,
            ]
            for i, move in enumerate(state.moves)
        ]
    ).T
