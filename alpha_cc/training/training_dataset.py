import numpy as np
import torch
from torch.utils.data import Dataset

from alpha_cc.agents.mcts.mcts_agent import MCTSExperience


class TrainingDataset(Dataset):
    def __init__(self, trajectories: list[list[MCTSExperience]]) -> None:
        self._experiences = [exp for traj in trajectories for exp in traj]

    def __len__(self) -> int:
        return len(self._experiences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        exp = self._experiences[index]
        x = torch.as_tensor(exp.state.matrix).unsqueeze(0)
        value_target = torch.as_tensor(exp.v_target)
        pi_mask = torch.as_tensor(exp.state.action_mask)
        pi_target = self._create_pi_target_tensor(exp)
        return x.float(), value_target.float(), pi_target.float(), pi_mask.bool()

    def _create_pi_target_tensor(self, exp: MCTSExperience) -> torch.Tensor:
        """
        The neural net and the MCTS speak different lanugages:
        - nn thinks a policy is a big tensor encoding all moves (including legal ones)
        - mcts things a policy is a vector for only the legal moves

        This is because neural nets need a fixes size output with semantic consistency,
        and MCTS is more natural to write with illegal moves not even considered.

        This method translates mcts-style pi targets into nn pi targets.

        """
        # TODO: make pretty
        pi_target = np.zeros(4 * (exp.state.board.size,))
        for i in range(len(exp.state.children)):
            from_coord, to_coord = exp.state.action_mask_indices[i]
            pi_target[from_coord.x, from_coord.y, to_coord.x, to_coord.y] = exp.pi_target[i]
        return torch.as_tensor(pi_target)