import click
import torch

from alpha_cc.agents import GreedyAgent, MCTSAgent
from alpha_cc.engine import Board
from alpha_cc.runtimes.runtime import RunTime, RunTimeConfig


@click.command("eval weights")
@click.argument("weights", type=click.Path(exists=True, dir_okay=False))
@click.option("--size", type=int, default=9)
@click.option("--n-rollouts", type=int, default=100)
@click.option("--rollout-depth", type=int, default=100)
@click.option("--training", is_flag=True)
@click.option("--vs-greedy", is_flag=True)
def main(
    weights: str,
    size: int,
    n_rollouts: int,
    rollout_depth: int,
    training: bool,
    vs_greedy: bool,
) -> None:
    board = Board(size)
    greedy_agent = GreedyAgent(size)
    agent = MCTSAgent(size, n_rollouts=n_rollouts, rollout_depth=rollout_depth)
    agent.nn.load_state_dict(torch.load(weights))
    agent.nn.eval()
    agents = (
        greedy_agent if vs_greedy else agent,
        agent,
    )
    config = RunTimeConfig(
        verbose=True,
        render=True,
    )
    runtime = RunTime(board, agents, config=config)
    winner = runtime.play_game(training=training)
    click.echo(f"Winner: {winner}")


if __name__ == "__main__":
    main()
