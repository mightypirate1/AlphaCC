import numpy as np

from alpha_cc.agents import MCTSAgent
from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.agents.value_assignment import ValueAssignmentStrategy
from alpha_cc.engine import Board
from alpha_cc.state import GameState


class TrainingRunTime:
    def __init__(
        self,
        board: Board,
        agent: MCTSAgent,
        value_assignment_strategy: ValueAssignmentStrategy,
    ) -> None:
        self._board = board
        self._agent = agent
        self._value_assignment_strategy = value_assignment_strategy

    def play_game(self, max_game_length: int | None = None, action_temperature: float = 1.0) -> list[MCTSExperience]:
        board = self._board.reset()
        agent = self._agent
        max_game_duration = np.inf if max_game_length is None else max_game_length
        agent.on_game_start()

        trajectory: list[MCTSExperience] = []
        while not board.info.game_over and board.info.duration < max_game_duration:
            pi, value = agent.run_rollouts(board, temperature=action_temperature)
            trajectory.append(
                MCTSExperience(
                    state=GameState(board),
                    pi_target=pi,
                    v_target=value,
                )
            )
            a = np.random.choice(len(pi), p=pi)
            move = board.get_moves()[a]
            board = board.apply(move)
        agent.on_game_end()
        return self._value_assignment_strategy(trajectory, final_board=board)
