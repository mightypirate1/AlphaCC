import numpy as np

from alpha_cc.agents.agent import Agent
from alpha_cc.agents.mcts.mcts import MCTS
from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.state import GameState
from alpha_cc.engine import Board


class MCTSAgent(MCTS, Agent):
    def __init__(self, board_size: int, max_game_length: int, n_rollouts: int = 100) -> None:
        super().__init__(board_size, max_game_length)
        self._n_rollouts = 2
        self._dirichlet_weight = 0.0  # 0.25  # TODO: get good value from paper
        self._trajectory: list[MCTSExperience] = []
        self._n_rollouts = n_rollouts

    @property
    def trajectory(self) -> list[MCTSExperience]:
        return self._trajectory

    def choose_move(self, board: Board, training: bool = False) -> int | np.integer:
        state = GameState(board)
        for _ in range(self._n_rollouts):
            v = self.rollout(state)
        pi = self.pi(state)

        if training:
            self._trajectory.append(
                MCTSExperience(
                    state=state,
                    pi_target=pi,
                    v_target=v,  # will be reassigned in `on_game_end`
                )
            )
            pi = self._training_policy(pi)
            return np.random.choice(len(pi), p=pi)
        return pi.argmax()

    def on_game_start(self) -> None:
        self.clear_nodes()
        self._trajectory = []

    def on_game_end(self) -> None:
        final_state = self.trajectory[-1].state
        v = -1.0 if final_state.info.winner == final_state.info.current_player else 1.0
        for experience in reversed(self.trajectory):
            experience.v_target = v
            v *= -1.0

    def _training_policy(self, pi: np.ndarray) -> np.ndarray:
        if self._dirichlet_weight == 0:
            return pi
        # TODO: learn about this!
        dirichlet_noise = np.random.dirichlet(np.full_like(pi, 1 / len(pi)))
        return (1 - self._dirichlet_weight) * pi + self._dirichlet_weight * dirichlet_noise
