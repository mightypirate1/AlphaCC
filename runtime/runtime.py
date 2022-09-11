import time
from dataclasses import dataclass
from typing import Optional, Tuple
from alpha_cc import Board, BoardInfo
from agents.base_agent import BaseAgent

@dataclass
class RunTimeConfig:
    verbose: bool = False
    render: bool = False
    slow: bool = False  # Does nothing if render is False
    debug: bool = False  # Currently doesn't do anything

class RunTime:
    def __init__(
        self,
        board: Board,
        agents: Tuple[BaseAgent,
        BaseAgent],
        config: Optional[RunTimeConfig] = None,
    ):
        self.board = board
        self.agent_dict = dict(enumerate(agents, start=1))
        self.config = config or RunTimeConfig()

    def play_game(self) -> None:
        ### Initialize
        self._agents_on_game_start()
        board = self.board.reset()
        game_over = False
        ### Play!
        while not game_over:
            agent = self.agent_dict[board.get_board_info().current_player]
            move = agent.choose_move(board)
            board = board.perform_move(move)
            game_over = (winner := board.get_board_info().winner > 0)
            ### Render?
            if self.config.render:
                board.render()
                if self.config.slow:
                    time.sleep(1)
        ### Be done
        if self.config.verbose:
            print(f"Player {winner} wins!")
        self._agents_on_game_end()

    def _agents_on_game_start(self) -> None:
        for _, agent in self.agent_dict.items():
            agent.on_game_start()

    def _agents_on_game_end(self) -> None:
        for _, agent in self.agent_dict.items():
            agent.on_game_end()
