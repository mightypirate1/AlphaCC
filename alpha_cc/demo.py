import torch

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.engine import Board
from alpha_cc.runtime.runtime import RunTime, RunTimeConfig

size = 5
starting_player = 2
board = Board(size)

agent = MCTSAgent(size, n_rollouts=100, rollout_depth=100)
agent.nn.load_state_dict(torch.load("tmp/models/dbg-1-size5/epoch-7.pth"))
agents = (
    agent,
    agent,
    # GreedyAgent(size),
    # GreedyAgent(size),
    # MCTSAgent(size, 150),
    # MCTSAgent(size, 150),
)
config = RunTimeConfig(
    verbose=True,
    render=True,
    slow=False,
    starting_player=starting_player,
)
runtime = RunTime(board, agents, config=config)
move_count = runtime.play_game(training=False)
print(f"Move count: {move_count}")  # noqa  # this file is just temporary
