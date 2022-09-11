import time
import alpha_cc
import numpy as np

from agents.heuristic_agents.dummy_agent import DummyAgent
from agents.heuristic_agents.greedy_agent import GreedyAgent

board = alpha_cc.Board(9)
agents = {
    1: DummyAgent(),
    2: GreedyAgent(),
}

for i in range(100000):
    info = board.get_board_info()
    current_player = info.current_player
    if info.winner != 0:
        print(f"{info.winner} wins!")
        break
    print(dict(player=current_player, agent=agents[current_player]))
    action = agents[current_player].choose_action(board)
    board = board.perform_move(action)
    board.render()
    time.sleep(1)
