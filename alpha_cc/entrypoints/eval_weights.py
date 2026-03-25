import click

from alpha_cc.agents import Agent, GreedyAgent
from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.agents.mcts.standalone_mcts_agent import StandaloneMCTSAgent
from alpha_cc.engine import Board
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
@click.option("--size", type=int, default=9)
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
    size: int,
    n_rollouts: int,
    rollout_depth: int,
    rollout_gamma: float,
    argmax_delay: int | None,
    training: bool,
    vs_greedy: bool,
    opponent: str | None,
    as_player_2: bool,
) -> None:
    def get_agent(path: str) -> StandaloneMCTSAgent:
        agent = StandaloneMCTSAgent(
            DefaultNet(size),
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
            return GreedyAgent(size)
        return get_agent(weights)

    _play_game(
        agents=(get_agent(weights), get_opponent()),
        size=size,
        as_player_2=as_player_2,
        training=training,
    )


@main.command()
@click.option("--nn-service-addr", type=str, required=True)
@click.option("--channel", type=int, default=0)
@click.option("--opponent-channel", type=int, default=None)
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=int, default=100)
@click.option("--rollout-depth", type=int, default=100)
@click.option("--rollout-gamma", type=float, default=1.0)
@click.option("--n-threads", type=int, default=1)
@click.option("--training", is_flag=True)
@click.option("--vs-greedy", is_flag=True)
@click.option("--as-player-2", is_flag=True)
def remote(
    nn_service_addr: str,
    channel: int,
    opponent_channel: int | None,
    size: int,
    n_rollouts: int,
    rollout_depth: int,
    rollout_gamma: float,
    n_threads: int,
    training: bool,
    vs_greedy: bool,
    as_player_2: bool,
) -> None:
    def get_agent(ch: int) -> MCTSAgent:
        return MCTSAgent(
            nn_service_addr=nn_service_addr,
            board_size=size,
            pred_channel=ch,
            n_rollouts=n_rollouts,
            rollout_depth=rollout_depth,
            rollout_gamma=rollout_gamma,
            n_threads=n_threads,
        )

    def get_opponent() -> Agent:
        if vs_greedy:
            return GreedyAgent(size)
        return get_agent(opponent_channel if opponent_channel is not None else channel)

    _play_game(
        agents=(get_agent(channel), get_opponent()),
        size=size,
        as_player_2=as_player_2,
        training=training,
    )
