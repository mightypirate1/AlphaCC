import logging
from pathlib import Path
from typing import Any

import click
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.config import Environment
from alpha_cc.db import TrainingDB
from alpha_cc.db.models import TournamentResult
from alpha_cc.entrypoints.logs import init_rootlogger
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.runtimes import TournamentRuntime
from alpha_cc.training import Trainer, TrainingDataset

logger = logging.getLogger(__file__)


@click.command("alpha-cc-trainer")
@click.option("--run-id", type=str, default="dbg-00")
@click.option("--size", type=int, default=9)
@click.option("--n-train-samples", type=int, default=1024)
@click.option("--epochs-per-update", type=int, default=1)
@click.option("--tournament-freq", type=int, default=10)
@click.option("--policy-weight", type=float, default=1.0)
@click.option("--value-weight", type=float, default=1.0)
@click.option("--entropy-weight", type=float, default=0.0)
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
    tournament_freq: int,
    policy_weight: float,
    value_weight: float,
    entropy_weight: float,
    batch_size: int,
    train_size: int,
    replay_buffer_size: int,
    lr: float,
    verbose: bool,
) -> None:
    init_rootlogger(verbose=verbose)
    db = TrainingDB(host=Environment.host_redis)
    replay_buffer = TrainingDataset(max_size=replay_buffer_size)
    summary_writer = create_summary_writer(run_id)
    trainer = Trainer(
        size,
        DefaultNet(size),
        epochs_per_update=epochs_per_update,
        policy_weight=policy_weight,
        value_weight=value_weight,
        entropy_weight=entropy_weight,
        batch_size=batch_size,
        lr=lr,
        summary_writer=summary_writer,
    )
    tournament_runtime = TournamentRuntime(size, db)

    db.flush_db()  # redis doesn't clear itself on restart currently...

    # workers will wait for the first weights getting published so everyone has the same net
    curr_index = db.weights_publish_latest(trainer.nn.state_dict())
    db.model_set_current(0, curr_index)
    n_iterations = 0
    champion_index = curr_index

    while True:
        # wait until we have enough new samples
        trajectories = await_samples(db, n_train_samples)
        logger.debug(f"fetched {len(trajectories)} trajectories")
        replay_buffer.add_trajectories(trajectories)
        trainer.report_rollout_stats(trajectories)

        # train on samples
        train_data = replay_buffer.sample(train_size)
        trainer.train(train_data)

        # publish weights
        curr_index = db.weights_publish_latest(trainer.nn.state_dict())
        db.model_set_current(0, curr_index)
        save_weights(run_id, curr_index, trainer.nn.state_dict())

        if (n_iterations := n_iterations + 1) % tournament_freq == 0:
            tournament_results = tournament_runtime.run_tournament([curr_index, champion_index], n_rounds=5)
            win_rate, win_rate_as_white, win_rate_as_black = extract_winrate(tournament_results)
            if win_rate > 0.55:
                champion_index = curr_index
                logger.info(f"new champion: {champion_index}! (winrate={win_rate})")
            log_winrate(win_rate, win_rate_as_white, win_rate_as_black, champion_index, n_iterations, summary_writer)


def await_samples(db: TrainingDB, n_train_samples: int) -> list[list[MCTSExperience]]:
    trajectories = []
    n_remaining = n_train_samples
    with tqdm(desc="awaiting samples", total=n_train_samples) as pbar:
        while n_remaining > 0:
            trajectory = db.trajectory_fetch(blocking=True)
            trajectories.append(trajectory)
            n_remaining -= len(trajectory)
            pbar.update(len(trajectory))
            pbar.set_postfix({"n": len(trajectory)})
    # if the trainer doesnt keep up with the workers, there will be samples
    # remaining on the queue. thus, we clear the queue so we can train on
    # the latest data. as as bonus, we also get to notice this  happening in
    # tensorboard
    if remaining_trajectories := db.trajectory_fetch_all():
        logger.warning(f"trainer is behind by {len(remaining_trajectories)} samples")
        trajectories.extend(remaining_trajectories)
    return trajectories


def extract_winrate(
    tournament_results: TournamentResult,
) -> tuple[float, float, float]:
    # assumes exactly two players were queued to play the tournament:
    #  1. current weights
    #  2. champion weights
    win_rate_as_white = tournament_results[1, 2]
    win_rate_as_black = 1 - tournament_results[2, 1]
    win_rate = (win_rate_as_white + win_rate_as_black) / 2
    return win_rate, win_rate_as_white, win_rate_as_black


def log_winrate(
    win_rate: float,
    win_rate_as_white: float,
    win_rate_as_black: float,
    champion_index: int,
    step: int,
    summary_writer: SummaryWriter,
) -> float:
    logger.info(f"WINRATES: total={win_rate} as_white={win_rate_as_white}, as_black={win_rate_as_black}")
    summary_writer.add_scalar("tournament/win-rate", win_rate, step)
    summary_writer.add_scalar("tournament/win-rate-as-white", win_rate_as_white, step)
    summary_writer.add_scalar("tournament/win-rate-as-black", win_rate_as_black, step)
    summary_writer.add_scalar("tournament/champion-index", champion_index, step)
    return win_rate


def save_weights(run_id: str, curr_index: int, weights: dict[str, Any]) -> None:
    path = f"{Environment.model_dir}/{run_id}/{str(curr_index).zfill(4)}.pth"
    latest_path = f"{Environment.model_dir}/{run_id}/latest.pth"
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(weights, latest_path)
    torch.save(weights, path)


def create_summary_writer(run_id: str) -> SummaryWriter:
    logdir = f"{Environment.tb_logdir}/{run_id}"
    Path(logdir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=logdir)
