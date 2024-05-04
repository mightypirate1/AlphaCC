import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent, MCTSExperience
from alpha_cc.engine import Board
from alpha_cc.training.trainer import Trainer
from alpha_cc.training.training_dataset import TrainingDataset


class StandaloneTrainer:  # TODO: remove this class when code stabilizes
    def __init__(
        self,
        agent: MCTSAgent,
        board: Board,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        epochs_per_update: int = 3,
        batch_size: int = 64,
        gamma: float = 1.0,
        gamma_delay: int | float = np.inf,
        lr: float = 1e-4,
        dataset_size: int = 100000,
        summary_writer: SummaryWriter | None = None,
    ) -> None:
        self._agent = agent
        self._board = board
        self._gamma = gamma
        self._gamma_delay = gamma_delay
        self._dataset = TrainingDataset(max_size=dataset_size)
        self._trainer = Trainer(
            board.info.size,
            agent.nn,
            policy_weight=policy_weight,
            value_weight=value_weight,
            epochs_per_update=epochs_per_update,
            batch_size=batch_size,
            gamma=gamma,
            gamma_delay=gamma_delay,
            lr=lr,
            summary_writer=summary_writer,
        )

    def train(self, num_samples: int = 1000, max_game_length: int | None = None) -> None:
        self._agent.nn.eval()
        trajectories: list[list[MCTSExperience]] = []
        with tqdm(desc="train rollouts", total=num_samples) as pbar:
            while num_samples > 0:
                trajectory = self.rollout_trajectory(max_game_length)
                trajectories.append(trajectory)
                self._dataset.add_trajectory(trajectory)
                n = len(trajectory)
                num_samples -= n
                pbar.update(n)
        self._trainer.report_rollout_stats(trajectories)
        self._trainer.train(self._dataset)

    @torch.no_grad()
    def rollout_trajectory(self, max_game_length: int | None = None) -> list[MCTSExperience]:
        def game_exceeds_duration() -> bool:
            if max_game_length is None:
                return False
            return board.info.duration >= max_game_length

        agent = self._agent
        board = self._board.reset()
        agent.on_game_start()

        while not board.info.game_over and not game_exceeds_duration():
            a = agent.choose_move(board, training=True)
            move = board.get_moves()[a]
            board = board.apply(move)

        self._agent.on_game_end(board)
        return self._agent.trajectory

    def _value_from_perspective_of_last_player(self, board: Board) -> float:
        if board.info.game_over:
            return -float(board.info.reward)  # minus because this board does not make it onto the trajectory
        return self._agent.trajectory[-1].v_target

    def _assign_value_targets(self, trajectory: list[MCTSExperience], value: float = 0.0) -> list[MCTSExperience]:
        for backwards_i, experience in enumerate(reversed(trajectory)):
            experience.v_target = value
            value *= -1.0 if backwards_i > self._gamma_delay else -self._gamma
        return trajectory
