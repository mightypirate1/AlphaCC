from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import Dataset

from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.state.game_state import GameState
from alpha_cc.state.state_tensors import state_tensor


class TrainingDataWeighter(ABC):
    @abstractmethod
    def weigh_internal_node(self, state: GameState, node: MCTSNodePy) -> float: ...


class TrainingDataset(Dataset):
    """
    Prioritized Experience Replay (PER) buffer using rank-based sampling.

    Ring buffer (deque) of MCTSExperience with parallel deques for KL-divergence
    and TD-error. New samples are initialized with max priority (visit-count
    gated for internal nodes). Sampling uses combined rank-based priorities:

        priority[i] = 1 / (rank_kl[i] * rank_td[i])^gamma

    After training, per-sample KL and TD errors are written back via
    update_priorities().
    """

    def __init__(
        self,
        experiences: Iterable[MCTSExperience] | None = None,
        max_size: int | None = None,
        gamma: float = 0.5,
        visits_threshold: float = 100.0,
        weighter: TrainingDataWeighter | None = None,
    ) -> None:
        if experiences is not None:
            exp_list = list(experiences)
            if max_size is None:
                max_size = len(exp_list)
        else:
            exp_list = []
            if max_size is None:
                max_size = 10000

        self._max_size = max_size
        self._gamma = gamma
        self._visits_threshold = visits_threshold
        self._weighter = weighter

        self._experiences: deque[MCTSExperience] = deque(maxlen=max_size)
        self._kl_div: deque[float] = deque(maxlen=max_size)
        self._td_error: deque[float] = deque(maxlen=max_size)

        for exp in exp_list:
            self._experiences.append(exp)
            self._kl_div.append(1.0)
            self._td_error.append(1.0)

    def __len__(self) -> int:
        return len(self._experiences)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        exp = self._experiences[index]

        x = state_tensor(exp.state)
        pi_mask = torch.as_tensor(exp.state.action_mask)
        pi_target = self._create_pi_target_tensor(exp)
        value_target = torch.as_tensor(exp.v_target)
        weight = torch.as_tensor(exp.weight)
        is_internal_node = torch.as_tensor(exp.is_internal_node)
        return (
            x.float(),
            pi_mask.bool(),
            pi_target.float(),
            value_target.float(),
            weight.float(),
            is_internal_node.bool(),
        )

    @property
    def samples(self) -> list[MCTSExperience]:
        return list(self._experiences)

    def prioritized_sample(self, n: int) -> tuple[np.ndarray, TrainingDataset]:
        """
        Rank-based PER sampling. Returns (buffer_indices, sampled_dataset).

        priority[i] = 1 / (rank_kl[i] * rank_td[i])^gamma
        where rank 1 = highest error (most surprising).
        """
        size = len(self._experiences)
        if n >= size:
            indices = np.arange(size)
            return indices, TrainingDataset(experiences=self.samples)

        kl = np.array(self._kl_div)
        td = np.array(self._td_error)
        rank_kl = rankdata(-kl, method="average")
        rank_td = rankdata(-td, method="average")
        priority = 1.0 / (rank_kl * rank_td) ** self._gamma
        probs = priority / priority.sum()
        indices = np.random.choice(size, size=n, replace=False, p=probs)
        sampled_exps = [self._experiences[i] for i in indices]
        return indices, TrainingDataset(experiences=sampled_exps)

    def update_priorities(self, indices: np.ndarray, kl_divs: np.ndarray, td_errors: np.ndarray) -> None:
        for i, idx in enumerate(indices):
            self._kl_div[idx] = float(kl_divs[i])
            self._td_error[idx] = float(td_errors[i])

    def split(self, frac: float) -> tuple[TrainingDataset, TrainingDataset]:
        all_samples = self.samples
        np.random.shuffle(all_samples)  # type: ignore
        n = int(len(all_samples) * frac)
        return (
            TrainingDataset(all_samples[:n], weighter=self._weighter),
            TrainingDataset(all_samples[n:], weighter=self._weighter),
        )

    def add_data(self, training_data: TrainingData) -> None:
        self.add_trajectory(training_data.trajectory)
        self.add_internal_nodes(training_data.internal_nodes)

    def add_datas(self, training_datas: list[TrainingData]) -> None:
        for training_data in training_datas:
            self.add_data(training_data)

    def add_trajectories(self, trajectories: list[list[MCTSExperience]]) -> None:
        for traj in trajectories:
            self.add_trajectory(traj)

    def add_trajectory(self, trajectory: list[MCTSExperience]) -> None:
        for exp in trajectory:
            self._add_experience(exp, priority_scale=1.0)

    def add_internal_node(self, state: GameState, node: MCTSNodePy) -> None:
        n_visits = int(np.sum(node.n))
        pi_target = np.ones_like(node.n) / len(node.n) if n_visits == 0 else node.n / n_visits
        exp = MCTSExperience(
            state=state,
            pi_target=pi_target,
            v_target=0.0,
            weight=self._weighter.weigh_internal_node(state, node) if self._weighter else 1.0,
            is_internal_node=True,
        )
        priority_scale = min(1.0, n_visits / self._visits_threshold) if self._visits_threshold > 0 else 1.0
        self._add_experience(exp, priority_scale=priority_scale)

    def add_internal_nodes(self, internal_nodes: dict[GameState, MCTSNodePy]) -> None:
        for state, node in internal_nodes.items():
            self.add_internal_node(state, node)

    # Implausibly high init values so new samples rank highest until measured.
    # td_error: values in [-1,1] so max error is 2.  kl_div: rarely exceeds ~10.
    _INIT_TD_ERROR = 2.0
    _INIT_KL_DIV = 10.0

    def _add_experience(self, exp: MCTSExperience, priority_scale: float = 1.0) -> None:
        self._experiences.append(exp)
        self._kl_div.append(self._INIT_KL_DIV * priority_scale)
        self._td_error.append(self._INIT_TD_ERROR * priority_scale)

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

    def __setstate__(self, state: dict) -> None:
        """Backward compat: convert old two-buffer format to deque-based PER."""
        if "_kl_div" in state and isinstance(state["_kl_div"], deque):
            # New format
            self.__dict__.update(state)
            return

        # Old format had _experiences (deque) and _new_experiences (list)
        old_main = list(state.get("_experiences", []))
        old_new = state.get("_new_experiences", [])
        all_experiences = old_new + old_main
        max_size = state.get("_max_size", 10000)

        self._max_size = max_size
        self._weighter = state.get("_weighter")
        self._gamma = 0.5
        self._visits_threshold = 100.0
        self._experiences = deque[MCTSExperience](maxlen=max_size)
        self._kl_div = deque[float](maxlen=max_size)
        self._td_error = deque[float](maxlen=max_size)
        for exp in all_experiences:
            self._experiences.append(exp)
            self._kl_div.append(1.0)
            self._td_error.append(1.0)
