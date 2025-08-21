from pathlib import Path
from typing import Any, Self

import numpy as np
import torch

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.agents.mcts.mcts_node_py import MCTSNodePy
from alpha_cc.agents.mcts.node_store import LocalNodeStore, NodeStore
from alpha_cc.engine import Board
from alpha_cc.nn.nets import DefaultNet
from alpha_cc.state import GameState


class StandaloneMCTSAgent(MCTSAgent):
    """
    Standalone implementation of the MCTSAgent.

    It is slower, but can be used without the need of a Redis server,
    and the nn-service.

    TODO: this sublcasses MCTSAgent, but it should not ideally.
    It is done so that the TrainingRunTime can be passed this (for testing purposes).
    If the runtime accepts this type, then the runtime would import torch, which is
    undesirable since then the worker thread would have to import it as well, which
    takes a lot of memory.
    """

    def __init__(
        self,
        nn: DefaultNet,
        n_rollouts: int = 100,
        rollout_depth: int = 500,
        rollout_gamma: float = 1.0,
        dirichlet_weight: float = 0.0,
        dirichlet_alpha: float = 0.03,
        argmax_delay: int | None = None,
        c_puct_init: float = 2.5,
        c_puct_base: float = 19652.0,
        node_store: NodeStore | None = None,
    ) -> None:
        self._nn = nn
        self._n_rollouts = n_rollouts
        self._rollout_gamma = rollout_gamma
        self._rollout_depth = rollout_depth
        self._dirichlet_weight = dirichlet_weight
        self._dirichlet_alpha = dirichlet_alpha
        self._argmax_delay = argmax_delay
        self._c_puct_init = c_puct_init
        self._c_puct_base = c_puct_base
        self._steps_left_to_argmax = argmax_delay or np.inf
        self._node_store = node_store if node_store is not None else LocalNodeStore()

    @property
    def nn(self) -> DefaultNet:
        return self._nn

    @property
    def node_store(self) -> NodeStore:
        return self._node_store

    @property
    def internal_nodes(self) -> dict[Board, MCTSNodePy]:
        return {board: self.node_store.get(board) for board in self.node_store.keys()}  # noqa: SIM118

    def on_game_start(self) -> None:
        self._steps_left_to_argmax = (self._argmax_delay or np.inf) + 1
        self.node_store.clear()

    def on_game_end(self) -> None:
        pass

    def choose_move(  # type: ignore
        self,
        board: Board,
        n_rollouts: int | None = None,
        rollout_depth: int | None = None,
        temperature: float = 1.0,
        training: bool = False,
    ) -> int:
        if self._argmax_delay is not None:
            self._steps_left_to_argmax -= 1
        pi, _ = self.run_rollouts(board, n_rollouts=n_rollouts, rollout_depth=rollout_depth, temperature=temperature)
        action_index = int(pi.argmax())
        if training and self._steps_left_to_argmax > 0:
            action_index = np.random.choice(len(pi), p=pi)
        return action_index

    def run_rollouts(
        self,
        board: Board,
        temperature: float = 1.0,
        n_rollouts: int | None = None,
        rollout_depth: int | None = None,
    ) -> tuple[np.ndarray, float]:
        n_rollouts = n_rollouts if n_rollouts is not None else self._n_rollouts
        rollout_depth = rollout_depth if rollout_depth is not None else self._rollout_depth
        state = GameState(board)
        value = -np.array([self._rollout(state, remaining_depth=rollout_depth) for _ in range(n_rollouts)]).mean()
        pi = self._rollout_policy(state.board, temperature)
        return pi, value

    def with_weights(self, path: str | Path) -> Self:
        weights = torch.load(path, weights_only=True)
        self.load_weights(weights)
        return self

    def load_weights(self, weights: dict[str, Any]) -> None:
        self.nn.load_state_dict(weights)

    def set_node_store(self, node_store: NodeStore) -> None:
        self._node_store = node_store

    def _rollout_policy(self, board: Board, temperature: float = 1.0) -> np.ndarray:
        node = self.node_store.get(board)
        weighted_counts = node.n
        if temperature != 1.0:  # save some flops
            weighted_counts = node.n ** (1 / temperature)
        pi = weighted_counts / weighted_counts.sum()
        return pi

    @torch.no_grad()
    def _rollout(self, state: GameState, remaining_depth: int = 999) -> float | np.floating:
        """
        recursive rollout that traverses the game-tree and updates nodes along the way.
        the return value is the value as seen by the parent node, hence all the minuses.
        """

        def add_as_new_node(pi: np.ndarray, v_hat: float | np.floating) -> None:
            if self._dirichlet_weight > 0.0:
                dirichlet_noise = np.random.dirichlet(self._dirichlet_alpha * pi)
                pi_noised = (1 - self._dirichlet_weight) * pi + self._dirichlet_weight * dirichlet_noise
                pi = pi_noised
            self.node_store.set(
                state.board,
                MCTSNodePy(
                    pi=pi,
                    v_hat=v_hat,
                    n=np.zeros(len(state.children), dtype=np.uint16),
                    q=np.zeros(len(state.children), dtype=np.float32),
                ),
            )

        # if game is over, we stop
        if state.info.game_over:
            return -state.info.reward

        # if we have reached as far as we have been (or depth is reached):
        # - (if depth not reached) initialize the node with nn estimates and zeros for N(s,a), and Q(s,a)
        # - return value from the perspective of the player on the previous move
        if state.board not in self.node_store or remaining_depth == 0:
            pi, v_hat = self.nn.evaluate_state(state)
            if remaining_depth > 0:
                add_as_new_node(pi, v_hat)
            return -v_hat

        node = self.node_store.get(state.board)
        a = self._find_best_action(state)
        s_prime = GameState(state.board.apply(state.board.get_moves()[a]))
        v = self._rollout_gamma * self._rollout(s_prime, remaining_depth=remaining_depth - 1)

        # update node
        node.q[a] = (node.n[a] * node.q[a] + v) / (node.n[a] + 1)
        node.n[a] += 1
        self.node_store.set(state.board, node)

        return -v

    def _find_best_action(self, state: GameState) -> int:
        def node_c_puct() -> float:  # TODO: look at this again
            return self._c_puct_init + np.log((sum_n + self._c_puct_base + 1) / self._c_puct_base)

        node = self.node_store.get(state.board)
        sum_n = sum(node.n)
        c_puct = node_c_puct()

        prior_weight = c_puct * np.sqrt(sum_n) / (1 + node.n)
        u = node.q + prior_weight * node.pi
        best_action = np.argmax(u).astype(int)

        return best_action
