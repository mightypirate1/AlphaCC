from pathlib import Path
from typing import Any

import click
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.config import Environmnet
from alpha_cc.db import TrainingDB
from alpha_cc.entrypoints.logs import init_rootlogger
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.training.trainer import Trainer


@click.command("alpha-cc-trainer")
@click.option("--run-id", type=str, default="dbg-00")
@click.option("--size", type=int, default=9)
@click.option("--n-train-samples", type=int, default=1024)
@click.option("--epochs-per-update", type=int, default=3)
@click.option("--policy-weight", type=float, default=1.0)
@click.option("--value-weight", type=float, default=1.0)
@click.option("--lr", type=float, default=1e-4)
@click.option("--batch-size", type=int, default=64)
@click.option("--silent", is_flag=True, default=False)
def main(
    run_id: str,
    size: int,
    n_train_samples: int,
    epochs_per_update: int,
    policy_weight: float,
    value_weight: float,
    batch_size: int,
    lr: float,
    silent: bool,
) -> None:
    init_rootlogger(verbose=not silent)
    nn = DefaultNet(size)
    db = TrainingDB(host=Environmnet.host_redis)
    trainer = Trainer(
        size,
        nn,
        epochs_per_update=epochs_per_update,
        policy_weight=policy_weight,
        value_weight=value_weight,
        batch_size=batch_size,
        lr=lr,
        summary_writer=create_summary_writer(run_id),
    )

    while True:
        curr_index = db.publish_latest_weights(nn.state_dict())
        save_weights(run_id, curr_index, nn.state_dict())
        trajectories: list[list[MCTSExperience]] = []
        remaining_samples = n_train_samples
        with tqdm(desc="awaiting trajectories", total=n_train_samples) as pbar:
            while remaining_samples > 0:
                trajectory = db.fetch_experiences(blocking=True)
                trajectories.append(trajectory)
                pbar.update(len(trajectory))
                remaining_samples -= len(trajectory)
        trainer.train(trajectories)


def save_weights(run_id: str, curr_index: int, weights: dict[str, Any]) -> None:
    path = f"{Environmnet.model_dir}/{run_id}/{str(curr_index).zfill(4)}.pth"
    latest_path = f"{Environmnet.model_dir}/{run_id}/latest.pth"
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(weights, latest_path)
    torch.save(weights, path)


def create_summary_writer(run_id: str) -> SummaryWriter:
    logdir = f"{Environmnet.tb_logdir}/{run_id}"
    Path(logdir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=logdir)
