from pathlib import Path

import click
import torch

from alpha_cc.agents import Agent, GreedyAgent
from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.agents.mcts.standalone_mcts_agent import StandaloneMCTSAgent
from alpha_cc.engine import Board, GameConfig
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.runtimes.runtime import RunTime, RunTimeConfig


@click.group("eval")
def main() -> None:
    pass


def _play_game(
    agents: tuple[Agent, Agent],
    size: int,
    as_player_2: bool,
    training: bool,
) -> None:
    board = Board(size)
    if as_player_2:
        agents = agents[::-1]
    config = RunTimeConfig(
        verbose=True,
        render=True,
    )
    runtime = RunTime(board, agents, config=config)
    winner = runtime.play_game(training=training)
    click.echo(f"Winner: {winner}")


@main.command()
@click.argument("weights", type=click.Path(exists=True, dir_okay=False))
@click.option("--game", type=str, default="cc:9")
@click.option("--n-rollouts", type=int, default=100)
@click.option("--rollout-depth", type=int, default=100)
@click.option("--rollout-gamma", type=float, default=1.0)
@click.option("--argmax-delay", type=int, default=None)
@click.option("--training", is_flag=True)
@click.option("--vs-greedy", is_flag=True)
@click.option("--opponent", type=click.Path(exists=True, dir_okay=False))
@click.option("--as-player-2", is_flag=True)
def local(
    weights: str,
    game: str,
    n_rollouts: int,
    rollout_depth: int,
    rollout_gamma: float,
    argmax_delay: int | None,
    training: bool,
    vs_greedy: bool,
    opponent: str | None,
    as_player_2: bool,
) -> None:
    config = GameConfig(game)
    def get_agent(path: str) -> StandaloneMCTSAgent:
        agent = StandaloneMCTSAgent(
            DefaultNet(config),
            rollout_gamma=rollout_gamma,
            n_rollouts=n_rollouts,
            rollout_depth=rollout_depth,
            argmax_delay=argmax_delay,
        )
        return agent.with_weights(path)

    def get_opponent() -> Agent:
        if opponent is not None:
            if vs_greedy:
                raise ValueError("Cannot specify both --opponent and --vs-greedy")
            return get_agent(opponent)
        if vs_greedy:
            return GreedyAgent(config.board_size)
        return get_agent(weights)

    _play_game(
        agents=(get_agent(weights), get_opponent()),
        size=config.board_size,
        as_player_2=as_player_2,
        training=training,
    )


@main.command()
@click.option("--nn-service-addr", type=str, required=True)
@click.option("--channel", type=int, default=0)
@click.option("--opponent-channel", type=int, default=None)
@click.option("--game", type=str, default="cc:9")
@click.option("--n-rollouts", type=int, default=100)
@click.option("--rollout-depth", type=int, default=100)
@click.option("--rollout-gamma", type=float, default=1.0)
@click.option("--n-threads", type=int, default=1)
@click.option("--pruning-tree", is_flag=True)
@click.option("--opponent-pruning-tree", is_flag=True)
@click.option("--training", is_flag=True)
@click.option("--vs-greedy", is_flag=True)
@click.option("--as-player-2", is_flag=True)
def remote(
    nn_service_addr: str,
    channel: int,
    opponent_channel: int | None,
    game: str,
    n_rollouts: int,
    rollout_depth: int,
    rollout_gamma: float,
    n_threads: int,
    pruning_tree: bool,
    opponent_pruning_tree: bool,
    training: bool,
    vs_greedy: bool,
    as_player_2: bool,
) -> None:
    def get_agent(ch: int, pruning: bool) -> MCTSAgent:
        return MCTSAgent(
            nn_service_addr=nn_service_addr,
            pred_channel=ch,
            n_rollouts=n_rollouts,
            rollout_depth=rollout_depth,
            rollout_gamma=rollout_gamma,
            n_threads=n_threads,
            pruning_tree=pruning,
        )

    config = GameConfig(game)
    def get_opponent() -> Agent:
        if vs_greedy:
            return GreedyAgent(config.board_size)
        opp_ch = opponent_channel if opponent_channel is not None else channel
        return get_agent(opp_ch, opponent_pruning_tree)

    _play_game(
        agents=(get_agent(channel, pruning_tree), get_opponent()),
        size=config.board_size,
        as_player_2=as_player_2,
        training=training,
    )


@main.command("export-onnx")
@click.argument("weights", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--game", type=str, default="cc:9")
@click.option("--batch-size", type=int, default=None, help="Fixed batch dimension. If omitted, batch dim is dynamic.")
def export_onnx(
    weights: str,
    output: str,
    size: int,
    batch_size: int | None,
) -> None:
    """Load a state dict and export as ONNX."""
    model = DefaultNet(size)
    state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    batch = batch_size or 1
    dummy = torch.zeros(batch, 2, size, size)
    dynamic_axes = (
        None
        if batch_size is not None
        else {
            "input": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        }
    )
    out_path = Path(output)
    torch.onnx.export(
        model,
        (dummy,),
        out_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
    )
    click.echo(f"Exported to {out_path} (batch_size={'dynamic' if batch_size is None else batch_size})")
