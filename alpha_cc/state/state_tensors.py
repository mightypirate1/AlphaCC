import torch

from alpha_cc.state.game_state import GameState


def state_tensor(state: GameState) -> torch.Tensor:
    """
    Default tensor representation of a state:
    - one channel per player
    - pieces marked as 1.0, rest as 0.0
    """
    return torch.stack(
        [
            torch.as_tensor(state.matrix == 1),
            torch.as_tensor(state.matrix == 2),
        ]
    ).float()


def states_tensor(states: list[GameState]) -> torch.Tensor:
    """
    Tensor representation of a list of states.

    NOTE: assumes at least one state is passed
    """
    batch_size = len(states)
    state_shape = state_tensor(states[0]).shape
    x = torch.empty(batch_size, *state_shape, pin_memory=True)
    for i, state in enumerate(states):
        x[i] = state_tensor(state)
    return x
