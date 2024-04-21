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
    starting_player: int | None = None


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
        if self._config.starting_player is not None:
            board = self._board.reset_with_starting_player(self._config.starting_player)
        move_count = 0

        ### Play!
        while not board.board_info.game_over:
            agent = self._agent_dict[board.board_info.current_player]
            move = agent.choose_move(board, training=training)
            board = board.perform_move(move)

            move_count += 1
            if self._config.render:
                board.render()
            if self._config.slow:
                time.sleep(1)
            if self._config.verbose:
                print(f"Move {move_count} played.")  # noqa

        ### Be done
        if self._config.verbose:
            print(f"Player {board.board_info.winner} wins!")  # noqa
        self._agents_on_game_end()
        return move_count

    def _agents_on_game_start(self) -> None:
        for _, agent in self._agent_dict.items():
            agent.on_game_start()

    def _agents_on_game_end(self) -> None:
        for _, agent in self._agent_dict.items():
            agent.on_game_end()
