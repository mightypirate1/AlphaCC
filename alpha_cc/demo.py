from alpha_cc.agents.heuristic_agents.greedy_agent import GreedyAgent
from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.engine import Board
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.runtime.runtime import RunTime, RunTimeConfig

size = 9
starting_player = 2
board = Board(size)

agents = (
    GreedyAgent(size),
    # GreedyAgent(size),
    MCTSAgent(DefaultNet()),
)
config = RunTimeConfig(
    verbose=True,
    render=True,
    slow=False,
    starting_player=starting_player,
)
runtime = RunTime(board, agents, config=config)
runtime.play_game(training=False)
