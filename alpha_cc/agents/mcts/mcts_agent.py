from pathlib import Path
from typing import Any, Self

from lru import LRU
import numpy as np
import torch

from alpha_cc.agents.agent import Agent
from alpha_cc.agents.mcts.mcts_node import MCTSNode
from alpha_cc.engine import Board
from alpha_cc.nn.nets import DualHeadEvaluator
from alpha_cc.state import GameState, StateHash


class MCTSAgent(Agent):
    def __init__(
        self,
        nn: DualHeadEvaluator,
        n_rollouts: int = 100,
        rollout_depth: int = 500,
        dirichlet_weight: float = 0.0,
        dirichlet_alpha: float = 0.03,
        node_cache_size: int = 1000000,
    ) -> None:
        self._nn = nn
        self._n_rollouts = n_rollouts
        self._rollout_depth = rollout_depth
        self._dirichlet_weight = dirichlet_weight
        self._dirichlet_alpha = dirichlet_alpha
        self._nodes: LRU[StateHash, MCTSNode] = LRU(size=node_cache_size)

    @property
    def nn(self) -> DualHeadEvaluator:
        return self._nn

    @property
    def nodes(self) -> dict[StateHash, MCTSNode]:
        return self._nodes

    def on_game_start(self) -> None:
        self.nodes.clear()

    def on_game_end(self) -> None:
        pass

    def choose_move(self, board: Board, training: bool = False, temperature: float = 1.0) -> int | np.integer:
        pi, _ = self.run_rollouts(board, temperature=temperature)
        if training:
            return np.random.choice(len(pi), p=pi)
        return pi.argmax()

    def run_rollouts(self, board: Board, temperature: float = 1.0) -> tuple[np.ndarray, float]:
        state = GameState(board)
        value = np.array(
            [-self._rollout(state, remaining_depth=self._rollout_depth) for _ in range(self._n_rollouts)]
        ).mean()
        pi = self._rollout_policy(state, temperature)
        return pi, value

    def with_weights(self, path: str | Path) -> Self:
        weights = torch.load(path)
        self.load_weights(weights)
        return self

    def load_weights(self, weights: dict[str, Any]) -> None:
        if not isinstance(self.nn, torch.nn.Module):
            raise ValueError(f"Can't load weights into non-torch.nn.Module object {self.nn.__class__.__name__}.")
        self.nn.clear_cache()
        self.nn.load_state_dict(weights)

    def _rollout_policy(self, state: GameState, temperature: float = 1.0) -> np.ndarray:
        node = self._nodes[state.hash]
        weighted_counts = node.n
        if temperature != 1.0:  # save some flops
            weighted_counts = node.n ** (1 / temperature)
        pi = weighted_counts / weighted_counts.sum()
        return pi

    @torch.no_grad()
    def _rollout(self, state: GameState, remaining_depth: int = 999) -> float:
        """
        recursive rollout that traverses the game-tree and updates nodes along the way.
        the return value is the value as seen by the parent node, hence all the minuses.
        """

        def add_as_new_node(pi: np.ndarray, v_hat: float | np.floating) -> None:
            if self._dirichlet_weight > 0.0:
                dirichlet_noise = np.random.dirichlet(self._dirichlet_alpha * pi)
                pi_noised = (1 - self._dirichlet_weight) * pi + self._dirichlet_weight * dirichlet_noise
                pi = pi_noised
            self._nodes[state.hash] = MCTSNode(
                pi=pi,
                v_hat=v_hat,
                n=np.zeros(len(state.children), dtype=np.uint16),
                q=np.zeros(len(state.children), dtype=np.float32),
            )

        # if game is over, we stop
        if state.info.game_over:
            return -state.info.reward

        # if we have reached as far as we have been (or depth is reached):
        # - (if depth not reached) initialize the node with nn estimates and zeros for N(s,a), and Q(s,a)
        # - return value from the perspective of the player on the previous move
        if state.hash not in self._nodes or remaining_depth == 0:
            v_hat = self.nn.value(state)
            pi = self.nn.policy(state)
            if remaining_depth > 0:
                add_as_new_node(pi, v_hat)
            return -v_hat

        node = self._nodes[state.hash]
        a = self._find_best_action(state)
        s_prime = GameState(state.board.apply(state.board.get_moves()[a]))
        v = self._rollout(s_prime, remaining_depth=remaining_depth - 1)

        # update node
        node.q[a] = (node.n[a] * node.q[a] + v) / (node.n[a] + 1)
        node.n[a] += 1

        return -v

    def _find_best_action(self, state: GameState) -> int:
        def node_c_puct() -> float:  # TODO: look at this again
            # according to some paper i forgot to reference...
            c_puct_init = 2.5
            c_puct_base = 19652
            return c_puct_init + np.log((sum_n + c_puct_base + 1) / c_puct_base)

        node = self._nodes[state.hash]
        sum_n = sum(node.n)
        c_puct = node_c_puct()

        prior_weight = c_puct * np.sqrt(sum_n) / (1 + node.n)
        u = node.q + prior_weight * node.pi
        best_action = np.argmax(u).astype(int)

        return best_action
