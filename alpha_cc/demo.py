from alpha_cc.agents.heuristic_agents.dummy_agent import DummyAgent
from alpha_cc.agents.heuristic_agents.greedy_agent import GreedyAgent
from alpha_cc.engine import Board
from alpha_cc.runtime.runtime import RunTime, RunTimeConfig

board = Board(5)
agents = (
    DummyAgent(),
    # GreedyAgent(),
    GreedyAgent(5),
)
config = RunTimeConfig(verbose=True, render=True, slow=True)
runtime = RunTime(board, agents, config=config)
runtime.play_game()
