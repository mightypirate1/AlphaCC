import numpy as np

from alpha_cc.agents.agent import Agent
from alpha_cc.agents.mcts.mcts import MCTS
from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.state import GameState
from alpha_cc.engine import Board


class MCTSAgent(MCTS, Agent):
    def __init__(self, board_size: int, n_rollouts: int = 100, rollout_depth: int = 999) -> None:
        super().__init__(board_size)
        self._n_rollouts = n_rollouts
        self._rollout_depth = rollout_depth
        self._dirichlet_weight = 0.0  # 0.25  # TODO: get good value from paper
        self._trajectory: list[MCTSExperience] = []

    @property
    def trajectory(self) -> list[MCTSExperience]:
        return self._trajectory

    def choose_move(self, board: Board, training: bool = False) -> int | np.integer:
        traversed_states = {exp.state.hash for exp in self.trajectory}
        state = GameState(board, disallowed_children=traversed_states)
        for _ in range(self._n_rollouts):
            v = self.rollout(state, remaining_depth=self._rollout_depth)
        pi = self.pi(state)

        if training:
            experience = MCTSExperience(state=state, pi_target=pi, v_target=v)
            self._trajectory.append(experience)
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
