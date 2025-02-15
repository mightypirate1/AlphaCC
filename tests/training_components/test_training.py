from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.training import Trainer, TrainingDataset


def test_training(
    training_dataset_with_content: TrainingDataset,
    trainer: Trainer,
    trajectory: list[MCTSExperience],
) -> None:
    trainer.train(
        training_dataset_with_content,
        len(training_dataset_with_content),
    )
    trainer.report_rollout_stats([trajectory])
