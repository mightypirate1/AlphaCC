from unittest.mock import patch

import pytest
from torch.utils.tensorboard import SummaryWriter

from alpha_cc.agents.mcts.node_store import LocalNodeStore
from alpha_cc.agents.mcts.standalone_mcts_agent import StandaloneMCTSAgent
from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.agents.value_assignment import (
    DefaultAssignmentStrategy,
)
from alpha_cc.engine import Board
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.runtimes import TrainingRunTime
from alpha_cc.training import Trainer, TrainingDataset

from .mocks import MockFileWriter


@pytest.fixture
def nn() -> DefaultNet:
    return DefaultNet(5)


@pytest.fixture
def agent(nn: DefaultNet) -> StandaloneMCTSAgent:
    return StandaloneMCTSAgent(nn, n_rollouts=10, rollout_depth=10, node_store=LocalNodeStore())


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
    return Trainer(5, nn, epochs_per_update=1, batch_size=8, summary_writer=summary_writer)


@pytest.fixture
def training_dataset() -> TrainingDataset:
    return TrainingDataset(max_size=128)


@pytest.fixture
def training_data(agent: StandaloneMCTSAgent, training_runtime: TrainingRunTime) -> TrainingData:
    return training_runtime.play_game(agent, max_game_length=8)


@pytest.fixture
def training_dataset_with_content(training_dataset: TrainingDataset, training_data: TrainingData) -> TrainingDataset:
    training_dataset.add_data(training_data)
    return training_dataset
