import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents import MCTSAgent
from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.engine import Board
from alpha_cc.runtimes.training_runtime import TrainingRunTime
from alpha_cc.training.trainer import Trainer
from alpha_cc.training.training_dataset import TrainingDataset


class StandaloneTrainer:
    def __init__(
        self,
        agent: MCTSAgent,
        board: Board,
        training_runtime: TrainingRunTime,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        epochs_per_update: int = 3,
        batch_size: int = 64,
        gamma: float = 1.0,
        gamma_delay: int | float = np.inf,
        lr: float = 1e-4,
        dataset_size: int = 10000,
        summary_writer: SummaryWriter | None = None,
    ) -> None:
        self._agent = agent
        self._board = board
        self._training_runtime = training_runtime
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
            lr=lr,
            summary_writer=summary_writer,
        )

    def train(self, num_samples: int = 1000, max_game_length: int | None = None) -> None:
        self._agent.nn.eval()
        trajectories: list[list[MCTSExperience]] = []
        with tqdm(desc="train rollouts", total=num_samples) as pbar:
            while num_samples > 0:
                trajectory = self._training_runtime.play_game(max_game_length=max_game_length)
                trajectories.append(trajectory)
                self._dataset.add_trajectory(trajectory)
                n = len(trajectory)
                num_samples -= n
                pbar.update(n)
        self._trainer.report_rollout_stats(trajectories)
        self._trainer.train(self._dataset)
