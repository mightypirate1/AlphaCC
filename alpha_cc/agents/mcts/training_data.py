from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alpha_cc.agents.mcts.mcts_experience import ProcessedExperience
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.worker_stats import WorkerStats
from alpha_cc.state import GameState


@dataclass
class SearchStatsAccumulator:
    """Per-move search diagnostics accumulated over a game."""

    prior_entropy: list[float] = field(default_factory=list)
    target_entropy: list[float] = field(default_factory=list)
    logit_std: list[float] = field(default_factory=list)
    sigma_q_std: list[float] = field(default_factory=list)
    kl_prior_posterior: list[float] = field(default_factory=list)
    kl_posterior_prior: list[float] = field(default_factory=list)

    def record(self, result: object) -> None:
        self.prior_entropy.append(result.prior_entropy)  # type: ignore[attr-defined]
        self.target_entropy.append(result.target_entropy)  # type: ignore[attr-defined]
        self.logit_std.append(result.logit_std)  # type: ignore[attr-defined]
        self.sigma_q_std.append(result.sigma_q_std)  # type: ignore[attr-defined]
        self.kl_prior_posterior.append(result.kl_prior_posterior)  # type: ignore[attr-defined]
        self.kl_posterior_prior.append(result.kl_posterior_prior)  # type: ignore[attr-defined]

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "prior-entropy": np.array(self.prior_entropy, dtype=np.float32),
            "target-entropy": np.array(self.target_entropy, dtype=np.float32),
            "logit-std": np.array(self.logit_std, dtype=np.float32),
            "sigma-q-std": np.array(self.sigma_q_std, dtype=np.float32),
            "kl-prior-posterior": np.array(self.kl_prior_posterior, dtype=np.float32),
            "kl-posterior-prior": np.array(self.kl_posterior_prior, dtype=np.float32),
        }


@dataclass
class TrainingData:
    trajectory: list[ProcessedExperience]
    internal_nodes: dict[GameState, MCTSNodePy]
    worker_stats: WorkerStats = field(default_factory=WorkerStats.empty)
    search_stats: SearchStatsAccumulator = field(default_factory=SearchStatsAccumulator)
    winner: int = 0  # 0 = draw/early, 1 = P1, 2 = P2

    def __bool__(self) -> bool:
        return bool(self.trajectory)
