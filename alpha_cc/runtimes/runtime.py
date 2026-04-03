import time

from alpha_cc.agents.agent import Agent
from alpha_cc.engine import Board
from alpha_cc.runtimes.runtime_config import RunTimeConfig


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
        board = self._board.reset()
        self._agents_on_game_start()

        max_len = self._config.max_game_length
        while not board.info.game_over:
            if max_len is not None and board.info.duration >= max_len:
                if self._config.verbose:
                    print(f"Draw! Max game length ({max_len}) reached.")  # noqa
                self._agents_on_game_end()
                return 0
            agent = self._agent_dict[board.info.current_player]
            agent_index = agent.choose_move(board, training=training)
            move = board.get_moves()[agent_index]
            board = board.apply(move)
            for a in self._agent_dict.values():
                a.on_move_applied(board)

            if self._config.render:
                board.render()
            if self._config.slow:
                time.sleep(1)
            if self._config.verbose:
                print(f"Move {board.info.duration} played.")  # noqa

        if self._config.verbose:
            print(f"Player {board.info.winner} wins!")  # noqa
        self._agents_on_game_end()
        return board.info.winner

    def _agents_on_game_start(self) -> None:
        for _, agent in self._agent_dict.items():
            agent.on_game_start()

    def _agents_on_game_end(self) -> None:
        for _, agent in self._agent_dict.items():
            agent.on_game_end()
