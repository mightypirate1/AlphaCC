from unittest.mock import patch

import pytest
from torch.utils.tensorboard import SummaryWriter

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.agents.value_assignment import (
    DefaultAssignmentStrategy,
)
from alpha_cc.engine import Board, GameConfig
from alpha_cc.nn.nets.default_net import DefaultNet

_TEST_CONFIG = GameConfig("cc:5")
from alpha_cc.runtimes import TrainingRunTime
from alpha_cc.training import Trainer, TrainingDataset

from .mocks import MockFileWriter


@pytest.fixture
def nn() -> DefaultNet:
    return DefaultNet(_TEST_CONFIG)


@pytest.fixture
def agent() -> MCTSAgent:
    return MCTSAgent("http://bogus:12345", n_rollouts=10, rollout_depth=10, dummy_preds=True)


@pytest.fixture
def training_runtime() -> TrainingRunTime:
    return TrainingRunTime(Board(5), DefaultAssignmentStrategy(0.99))


@pytest.fixture
@patch("torch.utils.tensorboard.writer.FileWriter", new=MockFileWriter)
def summary_writer() -> SummaryWriter:
    def mock_close(self: SummaryWriter) -> None:
        pass

    log_dir = "bogus/dir"
    sw = SummaryWriter(log_dir=log_dir)
    sw.close = mock_close  # type: ignore
    return sw


@pytest.fixture
def trainer(nn: DefaultNet, summary_writer: SummaryWriter) -> Trainer:
    return Trainer(_TEST_CONFIG, nn, epochs_per_update=1, batch_size=8, summary_writer=summary_writer)


@pytest.fixture
def training_dataset() -> TrainingDataset:
    return TrainingDataset(max_size=128)


@pytest.fixture
def training_data(agent: MCTSAgent, training_runtime: TrainingRunTime) -> TrainingData:
    return training_runtime.play_game(agent, n_rollouts=10, rollout_depth=10, max_game_length=8)


@pytest.fixture
def training_dataset_with_content(training_dataset: TrainingDataset, training_data: TrainingData) -> TrainingDataset:
    training_dataset.add_datas([training_data])
    return training_dataset
