import sys
from dataclasses import dataclass

import numpy as np

from alpha_cc.agents.state import GameState, StateHash
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.reward import HeuristicReward

# TODO: we need a stopping criterion, and to rule
# those cases before we hit a recursion limit
sys.setrecursionlimit(2000)


@dataclass
class Node:
    v_hat: np.floating
    pi: np.ndarray
    n: np.ndarray
    q: np.ndarray


class MCTS:
    def __init__(self, nn: DefaultNet) -> None:
        self._nn = nn
        self._nodes: dict[StateHash, Node] = {}

    @property
    def nn(self) -> DefaultNet:
        return self._nn

    @property
    def nodes(self) -> dict[StateHash, Node]:
        return self._nodes

    def clear_nodes(self) -> None:
        self._nodes.clear()

    def pi(self, state: GameState, temperature: float = 1.0) -> np.ndarray:
        node = self._nodes[state.hash]
        weighted_counts = node.n
        if temperature != 1.0:  # save some flops
            weighted_counts = node.n ** (1 / temperature)
        return weighted_counts / weighted_counts.sum()

    def rollout(self, state: GameState, fuse: int = 1000) -> np.floating | float:
        # if game is over, we stop
        if state.info.game_over:
            if state.info.winner == state.info.current_player:
                return 1.0  # previous player won
            return -1.0

        # if we have reached as far as we have been:
        # - initialize the node with nn estimates and zeros for N(s,a), and Q(s,a)
        # - return value from the perspective of the player on the previous move
        if state.hash not in self._nodes:
            v_hat = self._nn.value(state)
            pi = self._nn.policy(state)
            self._nodes[state.hash] = Node(
                v_hat=v_hat,
                pi=pi,
                n=np.zeros_like(pi, dtype=np.integer),
                q=np.zeros_like(pi),
            )
            return -v_hat

        if fuse == 0:
            # TODO: see todo at top of file. then we can remove the fuse
            heuristic = HeuristicReward(state.board.size)
            v_s = heuristic(state)
            v_sp = heuristic(state.children[1])
            return -(v_s - v_sp)

        # keep rolling
        a = self._find_best_action(state)
        s_prime = GameState(state.board.perform_move(a))
        v = self.rollout(s_prime, fuse=fuse - 1)

        # update node
        node = self._nodes[state.hash]
        node.q[a] = (node.n[a] * node.q[a] + v) / (node.n[a] + 1)
        node.n[a] += 1

        # return value from the perspective of the player on the previous move
        return -v

    def _find_best_action(self, state: GameState) -> int:
        def c_puct(node: Node) -> float:
            # according to the paper
            c_puct_init = 2.5
            c_puct_base = 19652
            return c_puct_init + np.log((np.sum(node.n) + c_puct_base + 1) / c_puct_base)

        best_u, best_a = -np.inf, -1
        node = self._nodes[state.hash]
        sum_n = sum(node.n)
        for a in range(len(state.board.get_all_possible_next_states())):
            q_sa = node.q[a]
            p_sa = node.pi[a]
            u = q_sa + c_puct(node) * p_sa * np.sqrt(sum_n / (1 + node.n[a]))
            if u > best_u:
                best_u = u
                best_a = a
        return best_a
