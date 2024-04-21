from pathlib import Path

import click
import torch
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.engine import Board
from alpha_cc.training.standalone_trainer import StandaloneTrainer


@click.command("train")
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=int, default=1000)
@click.option("--rollout-depth", type=int, default=20)
@click.option("--max-game-length", type=int, default=20)
@click.option("--n-train-samples", type=int, default=1024)
@click.option("--save-path", type=click.Path(dir_okay=True, file_okay=False), default=None)
@click.option("--init-weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--lr", type=float, default=1e-4)
def main(
    size: int,
    n_rollouts: int,
    rollout_depth: int,
    max_game_length: int,
    n_train_samples: int,
    save_path: str | None,
    init_weights: str | None,
    lr: float,
) -> None:
    if save_path is not None:
        model_dir = Path(save_path)
        model_dir.mkdir(parents=True, exist_ok=True)
    board = Board(size)
    agent = MCTSAgent(size, n_rollouts=n_rollouts, rollout_depth=rollout_depth)

    if init_weights is not None:
        agent.nn.load_state_dict(torch.load(init_weights))

    trainer = StandaloneTrainer(agent, board, lr=lr)
    for epoch in tqdm(range(20 * 8), desc="TRAINING"):
        trainer.train(n_train_samples, max_game_length=max_game_length)
        if save_path is not None:
            torch.save(agent.nn.state_dict(), model_dir / f"epoch-{str(epoch).zfill(4)}.pth")


if __name__ == "__main__":
    main()
