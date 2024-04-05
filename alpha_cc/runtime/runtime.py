import time

from alpha_cc_engine import Board
from pydantic import BaseModel

from alpha_cc.agents.base_agent import BaseAgent


class RunTimeConfig(BaseModel):
    verbose: bool = False
    render: bool = False
    slow: bool = False  # Does nothing if render is False
    debug: bool = False  # Currently doesn't do anything


class RunTime:
    def __init__(
        self,
        board: Board,
        agents: tuple[BaseAgent, BaseAgent],
        config: RunTimeConfig | None = None,
    ) -> None:
        self.board = board
        self.agent_dict = dict(enumerate(agents, start=1))
        self.config = config or RunTimeConfig()

    def play_game(self) -> int:
        ### Initialize
        self._agents_on_game_start()
        board = self.board.reset()
        game_over = False
        move_count = 0

        ### Play!
        while not game_over:
            agent = self.agent_dict[board.get_board_info().current_player]
            move = agent.choose_move(board)
            board = board.perform_move(move)
            board_info = board.get_board_info()
            game_over = board_info.game_over
            
            move_count += 1
            if self.config.render:
                board.render()
            if self.config.slow:
                time.sleep(1)

        ### Be done
        if self.config.verbose:
            print(f"Player {board_info.winner} wins!")  # noqa
        self._agents_on_game_end()
        return move_count

    def _agents_on_game_start(self) -> None:
        for _, agent in self.agent_dict.items():
            agent.on_game_start()

    def _agents_on_game_end(self) -> None:
        for _, agent in self.agent_dict.items():
            agent.on_game_end()
