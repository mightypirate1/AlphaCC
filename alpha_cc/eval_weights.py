import click
import torch

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.engine import Board
from alpha_cc.runtime.runtime import RunTime, RunTimeConfig


@click.command("eval weights")
@click.argument("weights", type=click.Path(exists=True, dir_okay=False))
@click.option("--size", type=int, default=9)
@click.option("--slow", is_flag=True, default=False)
@click.option("--n-rollouts", type=int, default=1000)
@click.option("--rollout-depth", type=int, default=20)
def main(weights: str, size: int, slow: bool, n_rollouts: int, rollout_depth: int) -> None:
    board = Board(size)
    agent = MCTSAgent(size, n_rollouts=n_rollouts, rollout_depth=rollout_depth)
    agent.nn.load_state_dict(torch.load(weights))
    agent.nn.eval()
    agents = (agent, agent)
    config = RunTimeConfig(
        verbose=True,
        render=True,
        slow=slow,
    )
    runtime = RunTime(board, agents, config=config)
    move_count = runtime.play_game(training=False)
    print(f"Move count: {move_count}")  # noqa  # this file is just temporary


if __name__ == "__main__":
    main()
