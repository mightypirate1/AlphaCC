from pathlib import Path

import click
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.engine import Board
from alpha_cc.training.standalone_trainer import StandaloneTrainer


@click.command("train")
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=int, default=1000)
@click.option("--rollout-depth", type=int, default=20)
@click.option("--max-game-length", type=int, default=None)
@click.option("--n-train-samples", type=int, default=1024)
@click.option("--epochs-per-update", type=int, default=3)
@click.option("--policy-weight", type=float, default=1.0)
@click.option("--value-weight", type=float, default=1.0)
@click.option("--save-path", type=click.Path(dir_okay=True, file_okay=False), default=None)
@click.option("--init-weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--logdir", type=click.Path(), default=None)
@click.option("--lr", type=float, default=1e-6)
def main(
    size: int,
    n_rollouts: int,
    rollout_depth: int,
    max_game_length: int | None,
    n_train_samples: int,
    epochs_per_update: int,
    policy_weight: float,
    value_weight: float,
    save_path: str | None,
    init_weights: str | None,
    logdir: str | None,
    lr: float,
) -> None:
    if save_path is not None:
        model_dir = Path(save_path)
        model_dir.mkdir(parents=True, exist_ok=True)

    summary_writer = None
    if logdir is not None:
        summary_writer = SummaryWriter(logdir)

    board = Board(size)
    agent = MCTSAgent(
        size,
        n_rollouts=n_rollouts,
        rollout_depth=rollout_depth,
        rollout_max_game_length=max_game_length,
    )

    if init_weights is not None:
        agent.nn.load_state_dict(torch.load(init_weights))

    trainer = StandaloneTrainer(
        agent,
        board,
        lr=lr,
        policy_weight=policy_weight,
        value_weight=value_weight,
        epochs_per_update=epochs_per_update,
        summary_writer=summary_writer,
    )
    for epoch in tqdm(range(20 * 8), desc="TRAINING"):
        trainer.train(n_train_samples, max_game_length=max_game_length)
        if save_path is not None:
            torch.save(agent.nn.state_dict(), model_dir / f"epoch-{str(epoch).zfill(4)}.pth")
            torch.save(agent.nn.state_dict(), model_dir / "latest.pth")


if __name__ == "__main__":
    main()
