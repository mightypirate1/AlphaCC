import time
import alpha_cc
import numpy as np

from agents.heuristic_agents.dummy_agent import DummyAgent
from agents.heuristic_agents.greedy_agent import GreedyAgent
from runtime.runtime import RunTime, RunTimeConfig

board = alpha_cc.Board(9)
agents = [
    # DummyAgent(),
    GreedyAgent(),
    GreedyAgent(),
]
config = RunTimeConfig(verbose=True, render=True, slow=True)
runtime = RunTime(board, agents, config=config)
runtime.play_game()
