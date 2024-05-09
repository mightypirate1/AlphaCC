import logging
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
from alpha_cc.training import Trainer, TrainingDataset

logger = logging.getLogger(__file__)


@click.command("alpha-cc-trainer")
@click.option("--run-id", type=str, default="dbg-00")
@click.option("--size", type=int, default=9)
@click.option("--n-train-samples", type=int, default=1024)
@click.option("--epochs-per-update", type=int, default=1)
@click.option("--policy-weight", type=float, default=1.0)
@click.option("--value-weight", type=float, default=1.0)
@click.option("--batch-size", type=int, default=64)
@click.option("--train-size", type=int, default=5000)
@click.option("--replay-buffer-size", type=int, default=20000)
@click.option("--lr", type=float, default=1e-4)
@click.option("--verbose", is_flag=True, default=False)
def main(
    run_id: str,
    size: int,
    n_train_samples: int,
    epochs_per_update: int,
    policy_weight: float,
    value_weight: float,
    batch_size: int,
    train_size: int,
    replay_buffer_size: int,
    lr: float,
    verbose: bool,
) -> None:
    def await_sufficient_samples() -> list[list[MCTSExperience]]:
        trajectories = []
        n_remaining = n_train_samples
        with tqdm(desc="awaiting samples", total=n_train_samples) as pbar:
            while n_remaining > 0:
                trajectory = db.fetch_trajectory(blocking=True)
                trajectories.append(trajectory)
                n_remaining -= len(trajectory)
                pbar.update(len(trajectory))
                pbar.set_postfix({"n": len(trajectory)})
        return trajectories

    init_rootlogger(verbose=verbose)
    db = TrainingDB(host=Environmnet.host_redis)
    replay_buffer = TrainingDataset(max_size=replay_buffer_size)
    summary_writer = create_summary_writer(run_id)
    trainer = Trainer(
        size,
        DefaultNet(size),
        epochs_per_update=epochs_per_update,
        policy_weight=policy_weight,
        value_weight=value_weight,
        batch_size=batch_size,
        lr=lr,
        summary_writer=summary_writer,
    )

    db.flush_db()  # redis doesn't clear itself on restart currently...

    # workers will wait for the first weights getting published so everyone has the same net
    curr_index = db.publish_latest_weights(trainer.nn.state_dict())

    while True:
        # wait until we have enough new samples
        trajectories = await_sufficient_samples()
        logger.debug(f"fetched {len(trajectories)} trajectories")
        replay_buffer.add_trajectories(trajectories)
        trainer.report_rollout_stats(trajectories)

        # train on samples
        train_data = replay_buffer.sample(train_size)
        trainer.train(train_data)

        # publish weights
        curr_index = db.publish_latest_weights(trainer.nn.state_dict())
        save_weights(run_id, curr_index, trainer.nn.state_dict())


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
