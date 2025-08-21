import numpy as np
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.agents.mcts.mcts_experience import MCTSExperience
from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.agents.value_assignment import ValueAssignmentStrategy
from alpha_cc.engine import Board
from alpha_cc.state import GameState


class TrainingRunTime:
    def __init__(
        self,
        board: Board,
        value_assignment_strategy: ValueAssignmentStrategy,
    ) -> None:
        self._board = board
        self._value_assignment_strategy = value_assignment_strategy

    def play_game(
        self,
        agent: MCTSAgent,
        n_rollouts: int | None = None,
        rollout_depth: int | None = None,
        action_temperature: float = 1.0,
        max_game_length: int | None = None,
        argmax_delay: int | None = None,
    ) -> TrainingData:
        board = self._board.reset()
        max_game_duration = np.inf if max_game_length is None else max_game_length
        time_to_argmax = argmax_delay if argmax_delay is not None else np.inf
        agent.on_game_start()

        trajectory: list[MCTSExperience] = []

        with tqdm(desc="training", total=max_game_length) as pbar:
            while not board.info.game_over:  # main terminaltion condition
                if board.info.duration >= max_game_duration:  # ended early termination condition
                    for exp in trajectory:
                        exp.game_ended_early = True
                    break

                pi, value = agent.run_rollouts(
                    board,
                    n_rollouts=n_rollouts,
                    rollout_depth=rollout_depth,
                    temperature=action_temperature,
                )
                experience = MCTSExperience(
                    state=GameState(board),
                    pi_target=pi,
                    v_target=value,
                )
                trajectory.append(experience)

                a = int(np.argmax(pi))
                if (time_to_argmax := time_to_argmax - 1) >= 0:
                    a = np.random.choice(len(pi), p=pi)

                moves = board.get_moves()
                board = board.apply(moves[a])
                pbar.update(1)
        training_data = TrainingData(
            trajectory=self._value_assignment_strategy(trajectory, final_board=board),
            internal_nodes={GameState(board): nodepy for board, nodepy in agent.internal_nodes.items()},
        )
        agent.on_game_end()
        return training_data
