import alpha_cc_engine

from alpha_cc.agents.heuristic_agents.dummy_agent import DummyAgent
from alpha_cc.agents.heuristic_agents.greedy_agent import GreedyAgent
from alpha_cc.runtime.runtime import RunTime, RunTimeConfig

board = alpha_cc_engine.Board(9)
agents = (
    DummyAgent(),
    # GreedyAgent(),
    GreedyAgent(),
)
config = RunTimeConfig(verbose=True, render=True, slow=True)
runtime = RunTime(board, agents, config=config)
runtime.play_game()
