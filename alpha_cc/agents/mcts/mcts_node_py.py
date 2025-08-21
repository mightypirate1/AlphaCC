from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

from alpha_cc.engine import MCTSNode


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

    @classmethod
    def from_node(cls: type[Self], node: MCTSNode) -> Self:
        return cls(
            pi=np.array(node.pi, dtype=np.float32),
            v_hat=float(node.v),
            n=np.array(node.n, dtype=np.int32),
            q=np.array(node.q, dtype=np.float32),
        )

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
