from alpha_cc.agents.heuristic_agents.greedy_agent import GreedyAgent
from alpha_cc.engine import Board
from alpha_cc.runtime.runtime import RunTime


def test_integration() -> None:
    board = Board(5)
    agents = (
        GreedyAgent(5),
        GreedyAgent(5),
    )
    runtime = RunTime(board, agents)
    runtime.play_game()
