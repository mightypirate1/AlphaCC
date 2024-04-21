from dataclasses import dataclass

import numpy as np
import torch

from alpha_cc.agents.state import GameState, StateHash
from alpha_cc.nn.nets.default_net import DefaultNet


@dataclass
class Node:
    v_hat: float
    pi: np.ndarray  # this is the nn-output; not mcts pi
    n: np.ndarray
    q: np.ndarray


class MCTS:
    def __init__(self, board_size: int) -> None:
        self._nn = DefaultNet(board_size)
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

    @torch.no_grad()
    def rollout(self, state: GameState, remaining_depth: int = 999) -> float:
        def add_as_new_node(v_hat: float, pi: np.ndarray) -> None:
            self._nodes[state.hash] = Node(
                v_hat=v_hat,
                pi=pi,
                n=np.zeros(len(state.children), dtype=np.integer),
                q=np.zeros(len(state.children)),
            )

        # if game is over, we stop
        if state.info.game_over:
            if state.info.winner == state.info.current_player:
                return 1.0  # previous player won
            return -1.0

        # at some point one has to stop (recursion limit, feasability, etc)
        if remaining_depth == 0:
            return -float(self._nn.value(state))

        # if we have reached as far as we have been:
        # - initialize the node with nn estimates and zeros for N(s,a), and Q(s,a)
        # - return value from the perspective of the player on the previous move
        if state.hash not in self._nodes:
            v_hat = float(self._nn.value(state))
            pi = self._nn.policy(state)
            add_as_new_node(v_hat, pi)
            return -v_hat

        # keep rolling
        a = self._find_best_action(state)
        s_prime = GameState(
            state.board.perform_move(a),
            disallowed_children={state.hash, *state.disallowed_children},
        )
        v = self.rollout(s_prime, remaining_depth=remaining_depth - 1)

        # update node
        node = self._nodes[state.hash]
        node.q[a] = (node.n[a] * node.q[a] + v) / (node.n[a] + 1)
        node.n[a] += 1

        # return value from the perspective of the player on the previous move
        return -v

    def _find_best_action(self, state: GameState) -> int:
        def c_puct(node: Node) -> float:  # TODO: look at this again
            # according to some paper i forgot to reference...
            c_puct_init = 2.5
            c_puct_base = 19652
            return c_puct_init + np.log((node.n.sum() + c_puct_base + 1) / c_puct_base)

        best_u, best_a = -np.inf, -1
        node = self._nodes[state.hash]
        sum_n = sum(node.n)

        for a, sp in enumerate(state.children):
            if sp.hash in state.disallowed_children:
                continue
            q_sa = node.q[a]
            p_sa = node.pi[a]
            u = q_sa + c_puct(node) * p_sa * np.sqrt(sum_n) / (1 + node.n[a])

            if u > best_u:
                best_u = u
                best_a = a

        if best_a == -1:
            raise ValueError("our disallowed children got us stuck?")

        return best_a
