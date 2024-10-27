from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCTSNodePy:
    pi: np.ndarray  # this is the nn-output; not mcts pi
    v_hat: float | np.floating
    n: np.ndarray
    q: np.ndarray

    @property
    def v(self) -> float:
        if len(self.q) == 0:
            return 0.0
        return float(self.q.max())

    def with_flipped_value(self) -> MCTSNodePy:
        return MCTSNodePy(
            pi=self.pi,
            v_hat=-self.v_hat,
            n=self.n,
            q=-self.q,
        )

    def as_sorted(self) -> MCTSNodePy:
        order = np.argsort(self.n)[::-1]
        return MCTSNodePy(
            pi=self.pi[order],
            v_hat=self.v_hat,
            n=self.n[order],
            q=self.q[order],
        )
