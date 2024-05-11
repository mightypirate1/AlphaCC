import click

from alpha_cc.agents import Agent, GreedyAgent, MCTSAgent
from alpha_cc.engine import Board
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.runtimes.runtime import RunTime, RunTimeConfig


@click.command("eval weights")
@click.argument("weights", type=click.Path(exists=True, dir_okay=False))
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=int, default=100)
@click.option("--rollout-depth", type=int, default=100)
@click.option("--training", is_flag=True)
@click.option("--vs-greedy", is_flag=True)
@click.option("--opponent", type=click.Path(exists=True, dir_okay=False))
@click.option("--as-player-2", is_flag=True)
def main(
    weights: str,
    size: int,
    n_rollouts: int,
    rollout_depth: int,
    training: bool,
    vs_greedy: bool,
    opponent: str | None,
    as_player_2: bool,
) -> None:
    def get_agent(path: str) -> MCTSAgent:
        agent = MCTSAgent(DefaultNet(size), n_rollouts=n_rollouts, rollout_depth=rollout_depth)
        return agent.with_weights(path)

    def get_opponent() -> Agent:
        if opponent is not None:
            if vs_greedy:
                raise ValueError("Cannot specify both --opponent and --vs-greedy")
            return get_agent(opponent)
        if vs_greedy:
            return GreedyAgent(size)
        return get_agent(weights)

    board = Board(size)
    agents: tuple[Agent, Agent] = (
        get_agent(weights),
        get_opponent(),
    )
    if as_player_2:
        agents = agents[::-1]
    config = RunTimeConfig(
        verbose=True,
        render=True,
    )
    runtime = RunTime(board, agents, config=config)
    winner = runtime.play_game(training=training)
    click.echo(f"Winner: {winner}")


if __name__ == "__main__":
    main()
