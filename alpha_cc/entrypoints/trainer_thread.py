import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Any

import click
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts import MCTSExperience
from alpha_cc.config import Environment
from alpha_cc.db import TrainingDB
from alpha_cc.logs import init_rootlogger
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.runtimes import TournamentRuntime
from alpha_cc.training import TournamentManager, Trainer, TrainingCheckpoint, TrainingDataset

logger = logging.getLogger(__file__)
shutdown_requested = threading.Event()
checkpoint_lock = threading.Lock()


@click.command("alpha-cc-trainer")
@click.option("--run-id", type=str, default="dbg-00")
@click.option("--size", type=int, default=9)
@click.option("--n-train-samples", type=int, default=1024)
@click.option("--epochs-per-update", type=int, default=1)
@click.option("--tournament-freq", type=int, default=10)
@click.option("--policy-weight", type=float, default=1.0)
@click.option("--value-weight", type=float, default=1.0)
@click.option("--entropy-weight", type=float, default=0.0)
@click.option("--l2-reg", type=float, default=1e-4)
@click.option("--batch-size", type=int, default=64)
@click.option("--train-size", type=int, default=5000)
@click.option("--replay-buffer-size", type=int, default=20000)
@click.option("--lr", type=float, default=1e-4)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--init-run-id", type=str, default=None)
@click.option("--init-weights-index", type=int, default=None)
@click.option("--init-champion-weight-index", type=int, default=None)
@click.option("--gpu", is_flag=True, default=False)
def main(
    run_id: str,
    size: int,
    n_train_samples: int,
    epochs_per_update: int,
    tournament_freq: int,
    policy_weight: float,
    value_weight: float,
    entropy_weight: float,
    l2_reg: float,
    batch_size: int,
    train_size: int,
    replay_buffer_size: int,
    lr: float,
    verbose: bool,
    init_run_id: str | None,
    init_weights_index: int | None,
    init_champion_weight_index: int | None,
    gpu: bool,
) -> None:
    init_rootlogger(verbose=verbose)
    summary_writer = create_summary_writer(run_id)
    device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
    db = TrainingDB(host=Environment.redis_host_main)
    tournament_runtime = TournamentRuntime(size, db)
    replay_buffer = TrainingDataset(max_size=replay_buffer_size)
    trainer = Trainer(
        size,
        DefaultNet(size),
        epochs_per_update=epochs_per_update,
        policy_weight=policy_weight,
        value_weight=value_weight,
        entropy_weight=entropy_weight,
        batch_size=batch_size,
        lr=lr,
        l2_reg=l2_reg,
        device=device,
        summary_writer=summary_writer,
    )

    db.flush_db()  # redis doesn't currently clear itself on restart.

    curr_index, champion_index = initialize_weights(
        run_id, db, trainer, init_run_id, init_weights_index, init_champion_weight_index
    )

    tournament_manager = TournamentManager(
        tournament_runtime=tournament_runtime,
        training_db=db,
        summary_writer=summary_writer,
        run_id=run_id,
        champion_index=champion_index,
    )
    create_and_register_signal_handler(
        run_id,
        trainer,
        db,
        tournament_manager,
        replay_buffer,
    )
    db.model_set_current(0, curr_index)

    while True:
        # wait until we have enough new samples
        trajectories = await_samples(db, n_train_samples)
        logger.debug(f"fetched {len(trajectories)} trajectories")
        replay_buffer.add_trajectories(trajectories)
        trainer.report_rollout_stats(trajectories)

        # train on samples
        trainer.train(replay_buffer, train_size)
        replay_buffer.move_new_to_main_buffer()

        # publish weights
        curr_index = db.weights_publish_latest(trainer.nn.state_dict())
        db.model_set_current(0, curr_index)
        save_weights(run_id, curr_index, trainer.nn.state_dict())

        # periodically run tournament
        if curr_index % tournament_freq == 0:
            tournament_manager.run_tournament(curr_index)


def await_samples(db: TrainingDB, n_train_samples: int) -> list[list[MCTSExperience]]:
    """
    Awaiting samples from the workers, and returns the list of trajectories.
    """
    trajectories = []
    n_remaining = n_train_samples
    with tqdm(desc="awaiting samples", total=n_train_samples) as pbar:
        while n_remaining > 0:
            trajectory = db.trajectory_fetch(blocking=True)
            trajectories.append(trajectory)
            n_remaining -= len(trajectory)
            pbar.set_postfix({"n": len(trajectory)})
            pbar.update(len(trajectory))
    # if the trainer doesnt keep up with the workers, there will be samples
    # remaining on the queue. thus, we clear the queue so we can train on
    # the latest data. as as bonus, we also get to notice this  happening in
    # tensorboard
    if remaining_trajectories := db.trajectory_fetch_all():
        logger.warning(f"trainer is behind by {len(remaining_trajectories)} samples")
        trajectories.extend(remaining_trajectories)
    return trajectories


def initialize_weights(
    run_id: str,
    db: TrainingDB,
    trainer: Trainer,
    init_run_id: str | None,
    init_weights_index: int | None,
    init_champion_weight_index: int | None,
) -> tuple[int, int]:
    """
    If the run_id is not new, i.e. it has been used before for training,
    we will reuse the state from which it was stopped.

    Both initial and chapion weights, and which run_id we take the starting
    checkpoint from can be overridden. This is useful for experimentation
    when you might realize you have to tweak the hyperparameters, but are
    not sure which one. This way, you can fearlessly tinker around and see
    what works on a separate run_id, without messing up the state of the
    original run_id.

    For all these configurations, we here load the right stuff from what
    has been persisted on disk, and rig the db so that the right weights
    end up in the right place.

    Notice that your weights

    Puts the weights in the db and returns the index of the weights.

    """

    checkpoint = load_saved_checkpoint(init_run_id) if init_run_id else load_saved_checkpoint(run_id)

    if checkpoint is None:
        # no checkpoint found, start from scratch
        curr_index = db.weights_publish_latest(trainer.nn.state_dict())
        champion_index = curr_index
        logger.info(f"starting from scratch: {curr_index=}")
        return curr_index, champion_index

    if init_champion_weight_index is not None and init_champion_weight_index != checkpoint.champion_index:
        # we need to do extra work to load the old weights
        logger.info(f"overriding champion weights with: {run_id=}, weight_index={init_champion_weight_index}")
        init_weights = load_weights(checkpoint.run_id, init_champion_weight_index)
        checkpoint.champion_state_dict = init_weights
        checkpoint.champion_index = init_champion_weight_index

    if init_weights_index is not None and init_weights_index != checkpoint.current_index:
        # we need to do extra work to load the old weights
        logger.info(f"overriding main weights with: {run_id=}, weight_index={init_weights_index}")
        init_weights = load_weights(checkpoint.run_id, init_weights_index)
        if checkpoint.champion_index > checkpoint.current_index:
            # we rename the init weights so be after the champion
            # to prevent the champions from being overwritten
            #
            # TODO: think about all the weird combinations, and
            # what the intuitive behavior is.
            init_weights_index = checkpoint.champion_index + 1
        checkpoint.model_state_dict = init_weights
        checkpoint.current_index = init_weights_index

    # publish the weights to the db so they are available to the workers
    if checkpoint.current_index != checkpoint.champion_index:
        db.weights_publish(checkpoint.champion_state_dict, checkpoint.champion_index)
    db.weights_publish(checkpoint.model_state_dict, checkpoint.current_index, set_latest=True)
    return checkpoint.current_index, checkpoint.champion_index


def load_saved_checkpoint(run_id: str) -> TrainingCheckpoint | None:
    """
    Loads the latest checkpoint from the given run_id.
    """
    checkpoint_path = save_latest_checkpoint_path(run_id)
    if not Path(checkpoint_path).exists():
        return None
    return TrainingCheckpoint.from_path(checkpoint_path, verbose=True)


def create_and_register_signal_handler(
    run_id: str,
    trainer: Trainer,
    training_db: TrainingDB,
    tournament_manager: TournamentManager,
    replay_buffer: TrainingDataset,
) -> None:
    """
    Creates a signal handler that saves the training checkpoint and
    terminates the program when a signal is received.
    """

    def save_checkpoint() -> None:
        current_index = training_db.weights_fetch_latest_index()
        save_path_latest = save_latest_checkpoint_path(run_id)
        save_path_current = save_checkpoint_path(run_id, current_index)
        checkpoint = TrainingCheckpoint(
            run_id,
            model_state_dict=trainer.nn.state_dict(),
            champion_state_dict=training_db.weights_fetch(tournament_manager.champion_index),
            optimizer_state_dict=trainer.optimizer.state_dict(),
            current_index=current_index,
            champion_index=tournament_manager.champion_index,
            replay_buffer=replay_buffer,
        )
        checkpoint.save(save_path_latest, verbose=True)
        checkpoint.save(save_path_current, verbose=True)

    def signal_handler(signum: int, _: Any) -> None:
        if shutdown_requested.is_set():
            logger.warning("Shutdown already requested, ignoring signal")
            return
        shutdown_requested.set()
        logger.info(f"Received signal {signum}, saving checkpoint...")
        with checkpoint_lock:
            save_checkpoint()
        logger.info("Checkpoint saved, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def save_weights(run_id: str, curr_index: int, weights: dict[str, Any]) -> None:
    path = save_path(run_id, curr_index)
    latest_path = save_path_latest(run_id)
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(weights, latest_path)
    torch.save(weights, path)


def load_weights(run_id: str, index: int) -> dict[str, Any]:
    path = save_path(run_id, index)
    if not Path(path).exists():
        raise FileNotFoundError(f"weights file {path} does not exist")
    return torch.load(path, weights_only=True)


def save_path_latest(run_id: str) -> str:
    return f"{save_root(run_id)}/weights-latest.pth"


def save_path(run_id: str, index: int) -> str:
    return f"{save_root(run_id)}/weights-{str(index).zfill(4)}.pth"


def save_root(run_id: str) -> str:
    return f"{Environment.model_dir}/{run_id}"


def save_latest_checkpoint_path(run_id: str) -> str:
    return f"{save_root(run_id)}/checkpoint-latest.pth"


def save_checkpoint_path(run_id: str, index: int) -> str:
    return f"{save_root(run_id)}/checkpoint-{str(index).zfill(4)}.pth"


def create_summary_writer(run_id: str) -> SummaryWriter:
    logdir = f"{Environment.tb_logdir}/{run_id}"
    Path(logdir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=logdir)
