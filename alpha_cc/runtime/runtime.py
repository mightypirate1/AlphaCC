import time
from dataclasses import dataclass

from alpha_cc.agents.agent import Agent
from alpha_cc.engine import Board


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
        agents: tuple[Agent, Agent],
        config: RunTimeConfig | None = None,
    ) -> None:
        self._board = board
        self._agent_dict = dict(enumerate(agents, start=1))
        self._config = config or RunTimeConfig()

    def play_game(self, training: bool = False) -> int:
        ### Initialize
        self._agents_on_game_start()
        board = self._board.reset()
        move_count = 0

        ### Play!
        while not board.info.game_over:
            agent = self._agent_dict[board.info.current_player]
            a = agent.choose_move(board, training=training)
            move = board.get_moves()[a]
            board = board.apply(move)

            move_count += 1
            if self._config.render:
                board.render()
            if self._config.slow:
                time.sleep(1)
            if self._config.verbose:
                print(f"Move {move_count} played.")  # noqa

        ### Be done
        if self._config.verbose:
            print(f"Player {board.info.winner} wins!")  # noqa
        self._agents_on_game_end(board)
        return move_count

    def _agents_on_game_start(self) -> None:
        for _, agent in self._agent_dict.items():
            agent.on_game_start()

    def _agents_on_game_end(self, board: Board) -> None:
        for _, agent in self._agent_dict.items():
            agent.on_game_end(board)
