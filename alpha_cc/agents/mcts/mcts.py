from dataclasses import dataclass

import numpy as np

from alpha_cc.agents.state import GameState, StateHash
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.reward import HeuristicReward


@dataclass
class Node:
    v_hat: np.floating
    pi: np.ndarray  # this is the nn-output; not mcts pi
    n: np.ndarray
    q: np.ndarray


class MCTS:
    def __init__(self, board_size: int, max_game_length: int) -> None:
        self._max_game_length = max_game_length
        self._nn = DefaultNet(board_size)
        self._heuristic = HeuristicReward(board_size, scale=0.01)
        self._nodes: dict[StateHash, Node] = {}

    @property
    def nn(self) -> DefaultNet:
        return self._nn

    @property
    def nodes(self) -> dict[StateHash, Node]:
        return self._nodes

    @property
    def max_game_length(self) -> int:
        return self._max_game_length

    def clear_nodes(self) -> None:
        self._nodes.clear()

    def pi(self, state: GameState, temperature: float = 1.0) -> np.ndarray:
        node = self._nodes[state.hash]
        weighted_counts = node.n
        if temperature != 1.0:  # save some flops
            weighted_counts = node.n ** (1 / temperature)
        return weighted_counts / weighted_counts.sum()

    def rollout(self, state: GameState) -> np.floating | float:
        # if game is over, we stop
        if state.info.game_over:
            if state.info.winner == state.info.current_player:
                return 1.0  # previous player won
            return -1.0

        # heuristic for long games (we dont have deepmind's resources here)
        if state.info.duration == self._max_game_length:
            v_s = self._heuristic(state)
            v_sp = self._heuristic(state.children[1])
            return -(v_s - v_sp)

        # if we have reached as far as we have been:
        # - initialize the node with nn estimates and zeros for N(s,a), and Q(s,a)
        # - return value from the perspective of the player on the previous move
        if state.hash not in self._nodes:
            v_hat = self._nn.value(state)
            pi = self._nn.policy(state)
            self._nodes[state.hash] = Node(
                v_hat=v_hat,
                pi=pi,
                n=np.zeros(len(state.children), dtype=np.integer),
                q=np.zeros(len(state.children)),
            )
            return -v_hat

        # keep rolling
        a = self._find_best_action(state)
        s_prime = GameState(state.board.perform_move(a))
        v = self.rollout(s_prime)

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
