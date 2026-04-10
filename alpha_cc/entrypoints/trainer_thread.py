import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Literal

import click
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.config import Environment
from alpha_cc.db import TrainingDB
from alpha_cc.logs import init_rootlogger
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.runtimes import TournamentRuntime
from alpha_cc.training import ExportThread, StatsThread, TournamentManager, Trainer, TrainingCheckpoint, TrainingDataset

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
@click.option("--per-gamma", type=float, default=0.5)
@click.option("--per-visits-threshold", type=float, default=100.0)
@click.option("--per-rank-mode", type=click.Choice(["td", "min", "prod"]), default="td")
@click.option("--gpu", is_flag=True, default=False)
@click.option("--n-blocks", type=int, default=6)
@click.option("--hidden-channels", type=int, default=128)
@click.option("--onnx-compiled-batch-size", type=int, default=None)
@click.option("--onnx-compiled-batch-size-secondary", type=int, default=None)
@click.option("--stats-gpu", is_flag=True, default=False)
@click.option("--num-terminal-before-nn", type=int, default=0)
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
    per_gamma: float,
    per_visits_threshold: float,
    per_rank_mode: Literal["td", "min", "prod"],
    gpu: bool,
    n_blocks: int,
    hidden_channels: int,
    onnx_compiled_batch_size: int | None,
    onnx_compiled_batch_size_secondary: int | None,
    num_terminal_before_nn: int,
    stats_gpu: bool,
) -> None:
    init_rootlogger(verbose=verbose)
    torch._dynamo.config.suppress_errors = True
    torch.set_float32_matmul_precision("high")
    _patch_fx_traceback()
    summary_writer = create_summary_writer(run_id)
    device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
    db = TrainingDB(host=Environment.redis_host_main)
    tournament_runtime = TournamentRuntime(size, db)
    trainer = Trainer(
        size,
        DefaultNet(size, n_blocks=n_blocks, hidden_channels=hidden_channels),
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

    curr_index, champion_index, existing_checkpoint = initialize_training(
        run_id,
        size,
        db,
        trainer,
        init_run_id,
        init_weights_index,
        init_champion_weight_index,
        onnx_compiled_batch_size,
        n_blocks=n_blocks,
        hidden_channels=hidden_channels,
    )
    if existing_checkpoint is None:
        db.flush_db()  # safe fresh start
        db.nn_warmup_init(num_terminal_before_nn)
        curr_index, onnx_payload = publish_weights(trainer.nn, db, size, onnx_compiled_batch_size)
        save_weights(run_id, curr_index, trainer.nn.state_dict(), onnx_payload)
        champion_index = curr_index
        replay_buffer = TrainingDataset(
            max_size=replay_buffer_size,
            gamma=per_gamma,
            rank_mode=per_rank_mode,
            visits_threshold=per_visits_threshold,
        )
    else:
        replay_buffer = existing_checkpoint.replay_buffer
        db.nn_warmup_set(existing_checkpoint.nn_warmup_counter)
        logger.info(f"Restored replay buffer from checkpoint: size={len(replay_buffer)}")

    tournament_manager = TournamentManager(
        tournament_runtime=tournament_runtime,
        training_db=db,
        summary_writer=summary_writer,
        run_id=run_id,
        champion_index=champion_index,
        model=DefaultNet(size, n_blocks=n_blocks, hidden_channels=hidden_channels),
        board_size=size,
        onnx_compiled_batch_size_secondary=onnx_compiled_batch_size_secondary,
    )
    create_and_register_signal_handler(
        run_id,
        trainer,
        db,
        tournament_manager,
        replay_buffer,
        onnx_compiled_batch_size,
    )
    db.model_set_current(0, curr_index)
    if gpu:
        trainer.compile(board_size=size, mode="max-autotune")

    stats_device = "cuda" if stats_gpu and torch.cuda.is_available() else "cpu"
    stats_trainer = Trainer(
        size,
        DefaultNet(size, n_blocks=n_blocks, hidden_channels=hidden_channels),
        epochs_per_update=epochs_per_update,
        policy_weight=policy_weight,
        value_weight=value_weight,
        entropy_weight=entropy_weight,
        batch_size=batch_size,
        lr=lr,
        l2_reg=l2_reg,
        device=stats_device,
        summary_writer=summary_writer,
    )
    stats_thread = StatsThread(stats_trainer, base_limit=n_train_samples)
    stats_thread.start()

    export_thread = ExportThread(
        model=DefaultNet(size, n_blocks=n_blocks, hidden_channels=hidden_channels),
        db=db,
        run_id=run_id,
        board_size=size,
        save_weights_fn=save_weights,
        onnx_compiled_batch_size=onnx_compiled_batch_size,
        summary_writer=summary_writer,
    )
    export_thread.start()

    set_service_healthy()

    while True:
        warmup_remaining = db.nn_warmup_get()
        if warmup_remaining < 0:
            logger.info(f"Warmup: {-warmup_remaining} more terminal games until workers use real NN")
        # wait until we have enough new samples
        training_datas = await_samples(db, n_train_samples)
        for td in training_datas:
            if td.winner != 0:
                db.nn_warmup_increment()
        replay_buffer.add_datas(
            training_datas,
            summary_writer=summary_writer,
            global_step=curr_index,
            expected_num_samples=n_train_samples,
        )

        # prioritized sampling
        sampled_indices, sampled_dataset = replay_buffer.prioritized_sample(
            train_size,
            summary_writer=summary_writer,
            global_step=curr_index,
        )

        # train on sampled dataset and compute per-sample errors for PER
        kl_divs, td_errors = trainer.train(sampled_dataset)
        replay_buffer.update_priorities(sampled_indices, kl_divs, td_errors)

        # snapshot weights and hand off to background threads
        curr_index = db.weights_incr_weights_index()
        state_dict_snapshot = {k: v.cpu().clone() for k, v in trainer.nn.state_dict().items()}
        stats_thread.submit(state_dict_snapshot, training_datas, curr_index)
        export_thread.submit(state_dict_snapshot, curr_index)

        # periodically run tournament
        if curr_index % tournament_freq == 0:
            export_thread.wait_idle()
            tournament_manager.run_tournament(curr_index)


def await_samples(db: TrainingDB, n_train_samples: int) -> list[TrainingData]:
    """
    Awaiting samples from the workers, and returns the list of trajectories.
    """
    training_datas = []
    n_remaining = n_train_samples
    with tqdm(desc="awaiting samples", total=n_train_samples) as pbar:
        while n_remaining > 0 and not shutdown_requested.is_set():
            training_data = db.training_data_fetch(blocking=True)
            training_datas.append(training_data)
            n_remaining -= len(training_data.trajectory)
            pbar.set_postfix({"n": len(training_data.trajectory)})
            pbar.update(len(training_data.trajectory))
    # if the trainer doesnt keep up with the workers, there will be samples
    # remaining on the queue. thus, we clear the queue so we can train on
    # the latest data.
    if remaining_training_datas := db.training_data_fetch_all():
        n_remaining = sum([len(t.trajectory) for t in remaining_training_datas])
        logger.warning(f"trainer is behind by {n_remaining} samples")
        training_datas.extend(remaining_training_datas)
    return training_datas


def _serialize_model(
    model: torch.nn.Module,
    board_size: int,
    compiled_batch_size: int | None = None,
) -> bytes:
    model.eval()
    device = next(model.parameters()).device
    batch = compiled_batch_size or 1
    dummy = torch.zeros(batch, 2, board_size, board_size, device=device)
    tmp_path = Path(Environment.model_dir) / "temp.onnx"
    dynamic_axes = (
        None
        if compiled_batch_size is not None
        else {
            "input": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        }
    )
    torch.onnx.export(
        model,
        (dummy,),
        tmp_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
    )
    with open(tmp_path, "rb") as f:
        payload = f.read()
    model.train()
    return payload


def publish_weights(
    model: torch.nn.Module,
    db: TrainingDB,
    board_size: int,
    compiled_batch_size: int | None = None,
) -> tuple[int, bytes]:
    payload = _serialize_model(model, board_size, compiled_batch_size)
    curr_idx = db.weights_incr_weights_index()
    db.weights_publish(payload, curr_idx, batch_size=compiled_batch_size, set_latest=True)
    db.model_set_current(0, curr_idx)
    logger.info(f"published weights {curr_idx} (batch_size={compiled_batch_size})")
    return curr_idx, payload


def initialize_training(
    run_id: str,
    size: int,
    db: TrainingDB,
    trainer: Trainer,
    init_run_id: str | None,
    init_weights_index: int | None,
    init_champion_weight_index: int | None,
    onnx_compiled_batch_size: int | None = None,
    n_blocks: int = 6,
    hidden_channels: int = 128,
) -> tuple[int, int, TrainingCheckpoint | None]:
    """
    If the run_id is not new, i.e. it has been used before for training,
    we will reuse the state from which it was stopped.

    Both initial and champion weights, and which run_id we take the starting
    checkpoint from can be overridden. This is useful for experimentation
    when you might realize you have to tweak the hyperparameters, but are
    not sure which one. This way, you can fearlessly tinker around and see
    what works on a separate run_id, without messing up the state of the
    original run_id.

    Loads the right checkpoint from disk, restores the trainer state,
    and publishes weights to Redis so the nn-service can load them.
    """

    checkpoint = load_saved_checkpoint(init_run_id) if init_run_id else load_saved_checkpoint(run_id)

    if checkpoint is None:
        # no checkpoint found — main() will flush_db and publish fresh weights
        return 0, 0, None

    if init_champion_weight_index is not None and init_champion_weight_index != checkpoint.champion_index:
        logger.info(f"overriding champion weights with: {run_id=}, weight_index={init_champion_weight_index}")
        init_weights = load_weights(checkpoint.run_id, init_champion_weight_index)
        # re-serialize the overridden champion as ONNX
        champion_model = DefaultNet(size, n_blocks=n_blocks, hidden_channels=hidden_channels)
        champion_model.load_state_dict(init_weights)
        checkpoint.champion_payload = _serialize_model(champion_model, size, onnx_compiled_batch_size)
        checkpoint.champion_index = init_champion_weight_index

    if init_weights_index is not None and init_weights_index != checkpoint.current_index:
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

    # set the weights/clocks in the trainer
    trainer.nn.load_state_dict(checkpoint.model_state_dict)
    trainer.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    trainer.set_steps(*checkpoint.trainer_steps)
    # publish the weights to redis so the nn-service can load them
    if checkpoint.current_index != checkpoint.champion_index:
        db.weights_publish(checkpoint.champion_payload, checkpoint.champion_index, batch_size=onnx_compiled_batch_size)
    current_payload = _serialize_model(trainer.nn, size, onnx_compiled_batch_size)
    db.weights_publish(current_payload, checkpoint.current_index, batch_size=onnx_compiled_batch_size, set_latest=True)
    return checkpoint.current_index, checkpoint.champion_index, checkpoint


def load_saved_checkpoint(run_id: str) -> TrainingCheckpoint | None:
    """
    Loads the latest checkpoint from the given run_id.
    """
    checkpoint_path = save_latest_checkpoint_path(run_id)
    if not Path(checkpoint_path).exists():
        return None
    return TrainingCheckpoint.from_path(checkpoint_path, verbose=True)


def set_service_healthy() -> None:
    Path("/tmp/healthy").write_text("ok")  # noqa: S108
    logger.info("Trainer ready — health file written: /tmp/healthy")


def create_and_register_signal_handler(
    run_id: str,
    trainer: Trainer,
    training_db: TrainingDB,
    tournament_manager: TournamentManager,
    replay_buffer: TrainingDataset,
    onnx_compiled_batch_size: int | None = None,
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
            champion_payload=training_db.weights_fetch(
                tournament_manager.champion_index, batch_size=onnx_compiled_batch_size
            ),
            optimizer_state_dict=trainer.optimizer.state_dict(),
            current_index=current_index,
            champion_index=tournament_manager.champion_index,
            replay_buffer=replay_buffer,
            trainer_steps=trainer.get_steps(),
            nn_warmup_counter=training_db.nn_warmup_get(),
        )
        Path(save_root(run_id)).mkdir(exist_ok=True, parents=True)
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


def save_weights(run_id: str, curr_index: int, weights: dict[str, Any], onnx_payload: bytes) -> None:
    path = save_path(run_id, curr_index)
    latest_path = save_path_latest(run_id)
    onnx_path = save_path(run_id, curr_index, ext="onnx")
    onnx_latest_path = save_path_latest(run_id, ext="onnx")
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(weights, latest_path)
    torch.save(weights, path)
    Path(onnx_latest_path).write_bytes(onnx_payload)
    Path(onnx_path).write_bytes(onnx_payload)


def load_weights(run_id: str, index: int) -> dict[str, Any]:
    path = save_path(run_id, index)
    if not Path(path).exists():
        raise FileNotFoundError(f"weights file {path} does not exist")
    return torch.load(path, weights_only=True)


def save_path_latest(run_id: str, ext: str = "pth") -> str:
    return f"{save_root(run_id)}/weights-latest.{ext}"


def save_path(run_id: str, index: int, ext: str = "pth") -> str:
    return f"{save_root(run_id)}/weights-{str(index).zfill(4)}.{ext}"


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


def _patch_fx_traceback() -> None:
    """Workaround for a PyTorch bug where fx_traceback.annotate's __exit__
    does `del current_meta["custom"]` but the key may already be absent,
    causing a spurious KeyError that crashes the training loop."""
    from contextlib import contextmanager

    import torch.fx.traceback as fxt

    @contextmanager
    def _safe_annotate(fields: dict):  # type: ignore[no-untyped-def]  # noqa: ANN202
        prior = fxt.current_meta.get("custom")
        fxt.current_meta["custom"] = fields
        try:
            yield
        finally:
            if prior is None:
                fxt.current_meta.pop("custom", None)
            else:
                fxt.current_meta["custom"] = prior

    fxt.annotate = _safe_annotate  # type: ignore
