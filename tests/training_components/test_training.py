from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.training import Trainer, TrainingDataset


def test_training(
    training_dataset_with_content: TrainingDataset,
    trainer: Trainer,
    training_data: TrainingData,
) -> None:
    trainer.train(
        training_dataset_with_content,
        len(training_dataset_with_content),
    )
    trainer.report_rollout_stats([training_data])
