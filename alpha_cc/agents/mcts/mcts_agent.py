from dataclasses import dataclass

import numpy as np

from alpha_cc.agents.base_agent import BaseAgent
from alpha_cc.agents.mcts.mcts import MCTS
from alpha_cc.agents.state import GameState
from alpha_cc.engine import Board
from alpha_cc.nn.nets.pretend_net import DefaultNet


@dataclass
class MCTSExperience:
    state: GameState
    pi: np.ndarray
    v_target: np.floating | float = 0.0


class MCTSAgent(MCTS, BaseAgent):
    def __init__(self, nn: DefaultNet) -> None:
        super().__init__(nn)
        self._n_rollouts = 100
        self._dirichlet_weight = 0.0  # 0.25  # TODO: get good value from paper
        self._trajectory: list[MCTSExperience] = []

    @property
    def trajectory(self) -> list[MCTSExperience]:
        return self._trajectory

    def choose_move(self, board: Board, training: bool = False) -> int | np.integer:
        state = GameState(board)
        for _ in range(self._n_rollouts):
            self.rollout(state)
        pi = self.pi(state)
        if training:
            self._trajectory.append(MCTSExperience(state=state, pi=pi))
            pi = self._training_policy(pi)
            return np.random.choice(len(pi), p=pi)
        return pi.argmax()

    def on_game_start(self) -> None:
        self.clear_nodes()
        self._trajectory = []

    def on_game_end(self) -> None:
        pass

    def _training_policy(self, pi: np.ndarray) -> np.ndarray:
        if self._dirichlet_weight == 0:
            return pi
        # TODO: learn about this!
        dirichlet_noise = np.random.dirichlet(np.full_like(pi, 1 / len(pi)))
        return (1 - self._dirichlet_weight) * pi + self._dirichlet_weight * dirichlet_noise
