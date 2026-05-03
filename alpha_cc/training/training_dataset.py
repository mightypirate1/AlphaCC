from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from alpha_cc.agents.mcts.mcts_experience import ProcessedExperience
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.state.game_state import GameState

# Implausibly high init values so new samples rank highest until measured.
# wdl_error: cross-entropy typically in [0, 2].  kl_div: rarely exceeds ~10.
_INIT_TD_ERROR = 2.0
_INIT_KL_DIV = 10.0


class TrainingDataWeighter(ABC):
    @abstractmethod
    def weigh_internal_node(self, state: GameState, node: MCTSNodePy) -> float: ...


class TrainingDataset(Dataset):
    """
    Prioritized Experience Replay (PER) buffer using rank-based sampling.

    Ring buffer (deque) of ProcessedExperience with parallel deques for KL-divergence
    and TD-error. New samples are initialized with max priority (visit-count
    gated for internal nodes). Sampling uses combined rank-based priorities:

        mode "prod": priority[i] = 1 / (rank_kl[i] * rank_td[i])^gamma
        mode "min": priority[i] = 1 / min(rank_kl[i], rank_td[i])^gamma
        mode "td": priority[i] = 1 / rank_td[i]^gamma

    After training, per-sample KL and TD errors are written back via
    update_priorities().
    """

    def __init__(
        self,
        experiences: Iterable[ProcessedExperience] | None = None,
        max_size: int | None = None,
        gamma: float = 0.5,
        visits_threshold: float = 100.0,
        rank_mode: Literal["prod", "min", "td"] = "td",
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
        self._rank_mode = rank_mode
        self._weighter = weighter

        self._experiences: deque[ProcessedExperience] = deque(maxlen=max_size)
        self._kl_div: deque[float] = deque(maxlen=max_size)
        self._td_error: deque[float] = deque(maxlen=max_size)
        self._total_num_samples = 0

        for exp in exp_list:
            self._total_num_samples += 1
            self._experiences.append(exp)
            self._kl_div.append(1.0)
            self._td_error.append(1.0)

    def __len__(self) -> int:
        return len(self._experiences)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        exp = self._experiences[index]
        board = exp.state.board

        x = torch.as_tensor(board.state_tensor()).reshape(-1, board.info.size, board.info.size).float()
        pi_mask = torch.as_tensor(board.policy_mask().astype(bool))
        pi_target = torch.as_tensor(board.scatter_policy(np.array(exp.pi_target, dtype=np.float32)))
        wdl_target = torch.as_tensor(exp.wdl_target)
        weight = torch.as_tensor(exp.weight)
        is_internal_node = torch.as_tensor(exp.is_internal_node)
        return (
            x.float(),
            pi_mask.bool(),
            pi_target.float(),
            wdl_target.float(),
            weight.float(),
            is_internal_node.bool(),
        )

    @property
    def samples(self) -> list[ProcessedExperience]:
        return list(self._experiences)

    @property
    def total_num_samples(self) -> int:
        return self._total_num_samples

    def prioritized_sample(
        self,
        n: int,
        summary_writer: SummaryWriter | None = None,
        global_step: int = 0,
    ) -> tuple[np.ndarray, TrainingDataset]:
        """
        Rank-based PER sampling. Returns (buffer_indices, sampled_dataset).
        """
        size = len(self._experiences)
        if n >= size:
            indices = np.arange(size)
            return indices, TrainingDataset(experiences=self.samples)

        kl = np.array(self._kl_div)
        td = np.array(self._td_error)
        weights = np.array([e.weight for e in self._experiences])
        priority = weights / (self._joined_rank(kl, td)) ** self._gamma
        probs = priority / priority.sum()
        indices = np.random.choice(size, size=n, replace=False, p=probs)
        sampled_exps = [self._experiences[i] for i in indices]

        if summary_writer is not None:
            self._log_per_stats(summary_writer, global_step, sampled_exps, priority, kl, td)

        return indices, TrainingDataset(experiences=sampled_exps)

    def _joined_rank(self, kl: np.ndarray, td: np.ndarray) -> np.ndarray:
        if self._rank_mode == "prod":
            rank_kl = rankdata(-kl, method="average")
            rank_td = rankdata(-td, method="average")
            return rank_kl * rank_td
        if self._rank_mode == "min":
            return np.minimum(rankdata(-kl, method="average"), rankdata(-td, method="average"))
        if self._rank_mode == "td":
            return rankdata(-td, method="average")
        raise ValueError(f"Invalid rank_mode: {self._rank_mode}")

    def _log_per_stats(
        self,
        writer: SummaryWriter,
        step: int,
        sampled_exps: list[ProcessedExperience],
        priority: np.ndarray,
        kl: np.ndarray,
        td: np.ndarray,
    ) -> None:
        wdl_targets = np.array([e.wdl_target for e in sampled_exps])
        is_internal = np.array([e.is_internal_node for e in sampled_exps])
        is_terminal = np.array([e.weight == 1.0 and not e.is_internal_node for e in sampled_exps])

        writer.add_histogram("per/wdl-target-value-sampled", wdl_targets[:, 0] - wdl_targets[:, 2], global_step=step)
        writer.add_scalar("per/frac-internal-sampled", is_internal.mean(), global_step=step)
        writer.add_scalar("per/frac-terminal-sampled", is_terminal.mean(), global_step=step)
        writer.add_histogram("per/priority", priority, global_step=step)
        writer.add_histogram("per/kl-div-buffer", kl, global_step=step)
        writer.add_scalar("per/kl-div-buffer-mean", kl.mean(), global_step=step)
        writer.add_histogram("per/td-error-buffer", td, global_step=step)

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

    def add_datas(
        self,
        training_datas: list[TrainingData],
        summary_writer: SummaryWriter | None = None,
        global_step: int = 0,
        expected_num_samples: int | None = None,
    ) -> int:
        """Add training data, optionally log stats. Returns total trajectory steps added."""

        def add_trajectories(trajectories: list[list[ProcessedExperience]]) -> None:
            for traj in trajectories:
                add_trajectory(traj)

        def add_trajectory(trajectory: list[ProcessedExperience]) -> None:
            for exp in trajectory:
                self._add_experience(exp, priority_scale=1.0)

        def add_data(training_data: TrainingData) -> None:
            add_trajectory(training_data.trajectory)
            self.add_internal_nodes(training_data.internal_nodes)

        count = 0
        for training_data in training_datas:
            count += len(training_data.trajectory)
            add_data(training_data)
        self._total_num_samples += count

        if summary_writer is not None:
            overflow = 0 if expected_num_samples is None else max(0, count - expected_num_samples)
            summary_writer.add_scalar("trainer/total-samples-seen", self._total_num_samples, global_step=global_step)
            summary_writer.add_scalar("trainer/overflow", overflow, global_step=global_step)
            summary_writer.add_scalar("trainer/samples-this-batch", count, global_step=global_step)

        return count

    def add_internal_node(self, state: GameState, node: MCTSNodePy) -> None:
        n_visits = int(np.sum(node.n))
        pi_target = np.ones_like(node.n) / len(node.n) if n_visits == 0 else node.n / n_visits
        exp = ProcessedExperience(
            state=state,
            pi_target=pi_target,
            wdl_target=(0.0, 1.0, 0.0),  # pure uncertainty (draw) for internal nodes
            weight=self._weighter.weigh_internal_node(state, node) if self._weighter else 1.0,
            is_internal_node=True,
        )
        priority_scale = min(1.0, n_visits / self._visits_threshold) if self._visits_threshold > 0 else 1.0
        self._add_experience(exp, priority_scale=priority_scale)

    def add_internal_nodes(self, internal_nodes: dict[GameState, MCTSNodePy]) -> None:
        for state, node in internal_nodes.items():
            self.add_internal_node(state, node)

    def _add_experience(self, exp: ProcessedExperience, priority_scale: float = 1.0) -> None:
        self._experiences.append(exp)
        self._kl_div.append(_INIT_KL_DIV * priority_scale)
        self._td_error.append(_INIT_TD_ERROR * priority_scale)
