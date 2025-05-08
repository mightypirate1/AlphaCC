from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from alpha_cc.agents.mcts import MCTSExperience


class TrainingDataset(Dataset):
    """
    Has 2 buffers, one for new samples and one main buffer. This is so that one
    can sample the dataset and be sure to see the most recent ones along with a
    random selection of "old" samples.

    - on `add_trajectories` and `add_trajectory`, experiences are added as "new
    - on `sample`, "new" experiments are sampled, and then if there's enough
      room left, "old" samples are added. All "new" samples are moved to "old"
      once they have been sampled once.

    """

    def __init__(self, experiences: Iterable[MCTSExperience] | None = None, max_size: int = 10000) -> None:
        self._max_size = max_size
        self._experiences = deque[MCTSExperience](maxlen=max_size)
        self._new_experiences: list[MCTSExperience] = [] if experiences is None else list(experiences)

    def __len__(self) -> int:
        return len(self._new_experiences) + len(self._experiences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_new = len(self._new_experiences)
        # faster than concatenating
        exp = self._new_experiences[index] if index < n_new else self._experiences[index - n_new]

        x = exp.state.tensor
        pi_mask = torch.as_tensor(exp.state.action_mask)
        pi_target = self._create_pi_target_tensor(exp)
        value_target = torch.as_tensor(exp.v_target)
        weight = torch.as_tensor(exp.weight)
        return x.float(), pi_mask.bool(), pi_target.float(), value_target.float(), weight.float()

    @property
    def samples(self) -> list[MCTSExperience]:
        return self._new_experiences + list(self._experiences)

    def sample(self, batch_size: int, replace: bool = True) -> TrainingDataset:
        """
        Two cases:
        1. we request less samples than what is in the new buffer -> we sample from the new buffer
        2. we request more/all samples than what is in the new buffer -> we sample all from the new buffer,
              and then sample the rest from the old buffer
        """
        if batch_size < len(self._new_experiences):
            sampled_experiences = np.random.choice(
                self._new_experiences,  # type: ignore
                batch_size,
                replace=replace,
            ).tolist()
            return TrainingDataset(experiences=sampled_experiences, max_size=self._max_size)

        dataset_sample = TrainingDataset(experiences=self._new_experiences, max_size=self._max_size)
        remaining = batch_size - len(dataset_sample)
        if remaining > 0:
            sampled_experiences = np.random.choice(
                self._experiences,  # type: ignore
                min(remaining, len(self._experiences)),
                replace=replace,
            ).tolist()
            dataset_sample.add_trajectory(sampled_experiences)
        return dataset_sample

    def split(self, frac: float) -> tuple[TrainingDataset, TrainingDataset]:
        samples = self.samples
        np.random.shuffle(samples)  # type: ignore
        n = int(len(samples) * frac)
        return TrainingDataset(samples[:n]), TrainingDataset(samples[n:])

    def add_trajectories(self, trajectories: list[list[MCTSExperience]]) -> None:
        self._new_experiences.extend([exp for traj in trajectories for exp in traj])

    def add_trajectory(self, trajectory: list[MCTSExperience]) -> None:
        self._new_experiences.extend(trajectory)

    def move_new_to_main_buffer(self) -> None:
        self._experiences.extendleft(self._new_experiences)
        self._new_experiences.clear()

    def _create_pi_target_tensor(self, exp: MCTSExperience) -> torch.Tensor:
        """
        The neural net and the MCTS speak different lanugages:
        - nn thinks a policy is a big tensor encoding all moves (including legal ones)
        - mcts things a policy is a vector for only the legal moves

        This is because neural nets need a fixes size output with semantic consistency,
        and MCTS is more natural to write with illegal moves not even considered.

        This method translates mcts-style pi targets into nn pi targets.

        """

        size = exp.state.board.info.size
        pi_target = np.zeros((size, size, size, size))
        for i in range(len(exp.state.moves)):
            from_coord, to_coord = exp.state.action_mask_indices[i]
            pi_target[from_coord.x, from_coord.y, to_coord.x, to_coord.y] = exp.pi_target[i]
        return torch.as_tensor(pi_target)
