from typing import Self

from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.api.io.base_io import BaseIO


class MCTSNodeIO(BaseIO):
    pi: list[float]
    v_hat: float
    n: list[int]
    q: list[float]
    v: float

    @classmethod
    def from_mcts_node(cls: type[Self], node: MCTSNodePy) -> Self:
        return cls(
            pi=node.pi.tolist(),
            v_hat=float(node.v_hat),
            n=node.n.tolist(),
            q=node.q.tolist(),
            v=node.v,
        )
