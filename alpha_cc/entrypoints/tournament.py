from __future__ import annotations

import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path

import click
import grpc
import torch
from rich.console import Console
from rich.table import Table
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.engine import Board
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.proto import predict_pb2, predict_pb2_grpc
from alpha_cc.runtimes.runtime import RunTime
from alpha_cc.runtimes.runtime_config import RunTimeConfig

# ---------------------------------------------------------------------------
# gRPC helpers
# ---------------------------------------------------------------------------

_MAX_MSG = 256 * 1024 * 1024  # 256 MB


def _grpc_addr(addr: str) -> str:
    """Strip http:// scheme if present — Python grpcio wants bare host:port."""
    return addr.removeprefix("http://").removeprefix("https://")


def _tonic_addr(addr: str) -> str:
    """Ensure http:// scheme — Rust tonic wants a full URI."""
    if not addr.startswith("http://") and not addr.startswith("https://"):
        return f"http://{addr}"
    return addr


def _management_stub(addr: str) -> predict_pb2_grpc.ManagementServiceStub:
    channel = grpc.insecure_channel(
        _grpc_addr(addr),
        options=[
            ("grpc.max_send_message_length", _MAX_MSG),
            ("grpc.max_receive_message_length", _MAX_MSG),
        ],
    )
    return predict_pb2_grpc.ManagementServiceStub(channel)


def _get_server_info(addr: str) -> predict_pb2.ServerInfoResponse:
    stub = _management_stub(addr)
    return stub.GetServerInfo(predict_pb2.ServerInfoRequest())


def _load_model(addr: str, channel_id: int, onnx_bytes: bytes, version: int = 0) -> predict_pb2.LoadModelResponse:
    stub = _management_stub(addr)
    return stub.LoadModel(
        predict_pb2.LoadModelRequest(
            channel_id=channel_id,
            onnx_bytes=onnx_bytes,
            version=version,
        )
    )


# ---------------------------------------------------------------------------
# ONNX conversion helper
# ---------------------------------------------------------------------------


def _ensure_onnx_bytes(path: Path, game_size: int, batch_size: int | None) -> bytes:
    """Return ONNX bytes.  Auto-converts PyTorch state dicts (.pth/.pt)."""
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        return path.read_bytes()

    if suffix in (".pth", ".pt"):
        model = DefaultNet(game_size)
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        batch = batch_size or 1
        dummy = torch.zeros(batch, 2, game_size, game_size)
        dynamic_axes = (
            None
            if batch_size is not None
            else {
                "input": {0: "batch"},
                "policy": {0: "batch"},
                "value": {0: "batch"},
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp:
            torch.onnx.export(
                model,
                (dummy,),
                tmp.name,
                input_names=["input"],
                output_names=["policy", "value"],
                dynamic_axes=dynamic_axes,
                opset_version=18,
                do_constant_folding=True,
                external_data=False,
            )
            return Path(tmp.name).read_bytes()

    raise click.BadParameter(f"Unsupported file type: {suffix} (expected .onnx, .pth, or .pt)")


# ---------------------------------------------------------------------------
# Tournament result types
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    white_channel: int
    black_channel: int
    winner: int  # 1 = white won, 2 = black won, 0 = draw/timeout


@dataclass
class TournamentResults:
    channels: list[int]
    results: list[MatchResult] = field(default_factory=list)

    def pairwise_record(self, white: int, black: int) -> tuple[int, int, int]:
        """Returns (wins, losses, draws) for `white` when playing as white vs `black`."""
        wins = losses = draws = 0
        for r in self.results:
            if r.white_channel == white and r.black_channel == black:
                if r.winner == 1:
                    wins += 1
                elif r.winner == 2:
                    losses += 1
                else:
                    draws += 1
        return wins, losses, draws

    def aggregate(self, channel: int) -> dict:
        total = wins = losses = draws = 0
        wins_w = losses_w = games_w = 0
        wins_b = losses_b = games_b = 0
        for r in self.results:
            if r.white_channel == channel:
                total += 1
                games_w += 1
                if r.winner == 1:
                    wins += 1
                    wins_w += 1
                elif r.winner == 2:
                    losses += 1
                    losses_w += 1
                else:
                    draws += 1
            elif r.black_channel == channel:
                total += 1
                games_b += 1
                if r.winner == 2:
                    wins += 1
                    wins_b += 1
                elif r.winner == 1:
                    losses += 1
                    losses_b += 1
                else:
                    draws += 1
        return {
            "games": total,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_pct": wins / total * 100 if total else 0,
            "win_pct_w": wins_w / games_w * 100 if games_w else 0,
            "win_pct_b": wins_b / games_b * 100 if games_b else 0,
        }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_pairwise_table(
    console: Console, results: TournamentResults, title: str, extract: Callable[[int, int, int], tuple[int, int]]
) -> None:
    table = Table(title=title)
    table.add_column("", style="bold")
    for ch in results.channels:
        table.add_column(f"ch{ch}", justify="center")

    for row_ch in results.channels:
        cells: list[str] = []
        for col_ch in results.channels:
            if row_ch == col_ch:
                cells.append("---")
            else:
                wins, losses, draws = results.pairwise_record(row_ch, col_ch)
                count, total = extract(wins, losses, draws)
                pct = count / total * 100 if total else 0
                cells.append(f"{pct:.0f}% ({count}/{total})")
        table.add_row(f"ch{row_ch}", *cells)

    console.print(table)


def _display_pairwise(console: Console, results: TournamentResults) -> None:
    _display_pairwise_table(
        console,
        results,
        "White Win Rate (row as White vs col as Black)",
        lambda wins, losses, draws: (wins, wins + losses + draws),
    )
    console.print()
    _display_pairwise_table(
        console,
        results,
        "Black Win Rate (row as Black vs col as White)",
        lambda wins, losses, draws: (losses, wins + losses + draws),
    )
    console.print()
    _display_pairwise_table(
        console,
        results,
        "Draw Rate (row as White vs col as Black)",
        lambda wins, losses, draws: (draws, wins + losses + draws),
    )


def _display_aggregate(console: Console, results: TournamentResults) -> None:
    table = Table(title="Aggregate Statistics")
    for col in ["Channel", "Games", "Wins", "Losses", "Draws", "Win%", "Win%(W)", "Win%(B)"]:
        table.add_column(col, justify="right" if col != "Channel" else "left")

    for ch in results.channels:
        agg = results.aggregate(ch)
        table.add_row(
            f"ch{ch}",
            str(agg["games"]),
            str(agg["wins"]),
            str(agg["losses"]),
            str(agg["draws"]),
            f"{agg['win_pct']:.1f}%",
            f"{agg['win_pct_w']:.1f}%",
            f"{agg['win_pct_b']:.1f}%",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Game playing
# ---------------------------------------------------------------------------


def _play_one_game(
    nn_service_addr: str,
    white_channel: int,
    black_channel: int,
    size: int,
    n_rollouts: int,
    rollout_depth: int,
    n_threads: int,
    max_game_length: int,
) -> MatchResult:
    tonic_addr = _tonic_addr(nn_service_addr)
    white_agent = MCTSAgent(
        nn_service_addr=tonic_addr,
        pred_channel=white_channel,
        n_rollouts=n_rollouts,
        rollout_depth=rollout_depth,
        n_threads=n_threads,
    )
    black_agent = MCTSAgent(
        nn_service_addr=tonic_addr,
        pred_channel=black_channel,
        n_rollouts=n_rollouts,
        rollout_depth=rollout_depth,
        n_threads=n_threads,
    )

    board = Board(size)
    config = RunTimeConfig(max_game_length=max_game_length)
    runtime = RunTime(board, (white_agent, black_agent), config=config)
    winner = runtime.play_game(training=False)

    return MatchResult(
        white_channel=white_channel,
        black_channel=black_channel,
        winner=winner,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group("tournament")
def main() -> None:
    """Tournament tool for comparing models served by nn-service."""


@main.command()
@click.option("--nn-service-addr", type=str, required=True, help="nn-service gRPC address (e.g. localhost:50055)")
def info(nn_service_addr: str) -> None:
    """Query nn-service for server info."""
    console = Console()
    resp = _get_server_info(nn_service_addr)

    console.print(f"\n[bold]Game size:[/bold] {resp.game_size}")
    console.print(f"[bold]Mode:[/bold] {'static' if resp.static_mode else 'redis-polling'}")

    table = Table(title="Channels")
    table.add_column("Channel", justify="right")
    table.add_column("Model Loaded", justify="center")
    table.add_column("Version", justify="right")

    for ch in resp.channels:
        table.add_row(
            str(ch.channel_id),
            "[green]Yes[/green]" if ch.model_loaded else "[red]No[/red]",
            str(ch.model_version) if ch.model_loaded else "-",
        )

    console.print(table)

    if resp.batch_sizes:
        bs_str = ", ".join(str(b) for b in resp.batch_sizes)
        console.print(f"[bold]Pipeline batch sizes:[/bold] {bs_str}")
    console.print()


@main.command("load-models")
@click.option("--nn-service-addr", type=str, required=True, help="nn-service gRPC address")
@click.option(
    "--model",
    type=(int, click.Path(exists=True)),
    multiple=True,
    help="channel_id and path to weights file (.onnx, .pth, .pt)",
)
@click.option("--batch-size", type=int, default=None, help="Fixed batch size for ONNX export (PyTorch weights only)")
def load_models(nn_service_addr: str, model: tuple[tuple[int, str], ...], batch_size: int | None) -> None:
    """Push ONNX models directly to nn-service channels."""
    if not model:
        raise click.UsageError("At least one --model is required")

    console = Console()
    server_info = _get_server_info(nn_service_addr)

    for channel_id, path_str in model:
        path = Path(path_str)
        console.print(f"Loading [bold]{path.name}[/bold] -> channel {channel_id} ... ", end="")
        onnx_bytes = _ensure_onnx_bytes(path, server_info.game_size, batch_size)
        resp = _load_model(nn_service_addr, channel_id, onnx_bytes)
        if resp.success:
            console.print("[green]OK[/green]")
        else:
            console.print(f"[red]FAILED[/red]: {resp.error}")


@main.command("run")
@click.option("--nn-service-addr", type=str, required=True, help="nn-service gRPC address")
@click.option(
    "--model",
    type=(int, click.Path(exists=True)),
    multiple=True,
    help="channel_id and path to weights file (.onnx, .pth, .pt)",
)
@click.option("--channels", type=str, default=None, help="Comma-separated channel IDs (if models already loaded)")
@click.option("--n-rounds", type=int, default=5, help="Number of tournament rounds")
@click.option("--n-rollouts", type=int, default=100, help="MCTS rollouts per move")
@click.option("--rollout-depth", type=int, default=100, help="MCTS rollout depth")
@click.option("--n-threads", type=int, default=1, help="MCTS threads per agent")
@click.option("--parallel-games", type=int, default=1, help="Number of games to play in parallel")
@click.option("--max-game-length", type=int, default=500, help="Max moves per game")
@click.option("--batch-size", type=int, default=None, help="Fixed batch size for ONNX export (PyTorch weights only)")
def run(
    nn_service_addr: str,
    model: tuple[tuple[int, str], ...],
    channels: str | None,
    n_rounds: int,
    n_rollouts: int,
    rollout_depth: int,
    n_threads: int,
    parallel_games: int,
    max_game_length: int,
    batch_size: int | None,
) -> None:
    """Run a round-robin tournament between models."""
    console = Console()
    server_info = _get_server_info(nn_service_addr)

    # Load models if provided
    if model:
        for channel_id, path_str in model:
            path = Path(path_str)
            console.print(f"Loading [bold]{path.name}[/bold] -> channel {channel_id} ... ", end="")
            onnx_bytes = _ensure_onnx_bytes(path, server_info.game_size, batch_size)
            resp = _load_model(nn_service_addr, channel_id, onnx_bytes)
            if resp.success:
                console.print("[green]OK[/green]")
            else:
                console.print(f"[red]FAILED[/red]: {resp.error}")
                raise click.Abort()

    # Determine channels
    if model:
        channel_ids = sorted({ch for ch, _ in model})
    elif channels:
        channel_ids = sorted(int(c.strip()) for c in channels.split(","))
    else:
        raise click.UsageError("Provide --model pairs or --channels")

    if len(channel_ids) < 2:
        raise click.UsageError("Need at least 2 channels for a tournament")

    # Build match schedule
    pairings = list(permutations(channel_ids, 2))
    total_games = len(pairings) * n_rounds

    console.print(f"\n[bold]Tournament:[/bold] {len(channel_ids)} channels, {n_rounds} rounds, {total_games} games")
    console.print(f"[bold]MCTS:[/bold] {n_rollouts} rollouts, depth={rollout_depth}, threads={n_threads}")
    console.print(f"[bold]Parallelism:[/bold] {parallel_games} concurrent games\n")

    schedule = []
    for _ in range(n_rounds):
        for white, black in pairings:
            schedule.append((white, black))

    results = TournamentResults(channels=channel_ids)

    game_size = server_info.game_size
    game_args = [
        (nn_service_addr, w, b, game_size, n_rollouts, rollout_depth, n_threads, max_game_length) for w, b in schedule
    ]

    with ProcessPoolExecutor(max_workers=parallel_games) as pool:
        futures = {pool.submit(_play_one_game, *args): args for args in game_args}
        for future in tqdm(as_completed(futures), total=total_games, desc="Games"):
            result = future.result()
            results.results.append(result)

    console.print()
    _display_pairwise(console, results)
    console.print()
    _display_aggregate(console, results)
    console.print()
