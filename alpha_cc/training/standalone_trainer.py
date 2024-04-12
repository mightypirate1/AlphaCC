from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent, MCTSExperience
from alpha_cc.engine import Board


class StandaloneTrainer:
    def __init__(
        self,
        agent: MCTSAgent,
        board: Board,
        gamma: float = 1.0,
    ) -> None:
        self._agent = agent
        self._board = board
        self._gamma = gamma

    def train(self, num_samples: int = 1000) -> None:
        trajectories: list[list[MCTSExperience]] = []
        with tqdm(desc="train rollouts", total=num_samples) as pbar:
            trajectory = self.rollout_trajectory()
            trajectories.append(trajectory)
            pbar.update(len(trajectory))
        self._update_nn(trajectories)

    def rollout_trajectory(self) -> list[MCTSExperience]:
        agent = self._agent
        board = self._board.reset()
        agent.on_game_start()

        while not board.board_info.game_over:
            move = agent.choose_move(board, training=True)
            board = board.perform_move(move)

        trajectory = agent.trajectory
        for i, experience in enumerate(reversed(trajectory)):
            experience.v_target = -self._gamma**i  # TODO: think about this

        self._agent.on_game_end()
        return trajectory

    def _update_nn(self, trajectories: list[list[MCTSExperience]]) -> None:
        self._agent.nn.update_weights(trajectories)
