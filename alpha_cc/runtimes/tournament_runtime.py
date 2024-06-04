import time
from logging import getLogger

import numpy as np
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.db.models import DBGameState, TournamentResult
from alpha_cc.db.training_db import TrainingDB
from alpha_cc.engine import Board
from alpha_cc.state import GameState

logger = getLogger(__file__)


class TournamentRuntime:
    """
    Class for arranging and runnining tournaments.
    - expects to be run across multiple services
    - currently supports only 2-player tournaments
    """

    def __init__(self, size: int, training_db: TrainingDB, max_game_length: int | None = None) -> None:
        self._size = size
        self._training_db = training_db
        self._max_game_duration = np.inf if max_game_length is None else max_game_length
        self._allocated_channels: list[int] | None = None  # used to help creator of tournament to clean up

    def play_and_record_game(self, agent_channel_dict: dict[int, MCTSAgent]) -> DBGameState:
        try:
            # since trainer will block until tournament is over,
            # we try-catch and post back the result in case of failure.
            # there is still a rare bug causing a crash

            board = Board(self._size)
            agents = tuple(agent_channel_dict.values())
            current_agent_idx = 0

            states = []
            actions = []
            with tqdm("tournament-game", total=self._max_game_duration) as pbar:
                while not board.info.game_over and board.info.duration < self._max_game_duration:
                    # get and record action/board
                    agent = agents[current_agent_idx]
                    action_index = agent.choose_move(board, training=False)
                    states.append(GameState(board))
                    actions.append(action_index)
                    # apply action
                    move = board.get_moves()[action_index]
                    board = board.apply(move)
                    pbar.update(1)
                    current_agent_idx = 1 - current_agent_idx
            player_1, player_2 = agent_channel_dict.keys()
            states.append(GameState(board))

        except Exception as e:
            logger.error("Tournament game FAILED!")
            # post back and hope for the best
            self._training_db.tournament_add_match(player_1, player_2)
            raise e

        # only player 1 wins are recorded
        # wins are counted this way to force decisive results
        if board.info.winner == 1:
            self._training_db.tournament_add_result(player_1, player_2, winner=player_1)
        elif board.info.winner == 2:
            self._training_db.tournament_add_result(player_1, player_2, winner=player_2)
        self._training_db.tournament_increment_counter()
        return DBGameState(states=states, move_idxs=actions, nodes={})

    def run_tournament(self, weight_indices: list[int], n_rounds: int = 5) -> TournamentResult:
        expected_games = self._arrange_tournament(weight_indices, n_rounds)
        return self._await_tournament_results(expected_games)

    def _arrange_tournament(self, weight_indices: list[int], n_rounds: int = 5) -> int:
        """
        Arranges a tournament by:
        - setting the model configuration to the given weight indices
        - adding matches to the tournament queue

        Returns the expected number of games

        """
        # assign weights to tournament channels
        for channel, weight_index in enumerate(weight_indices, start=1):
            self._training_db.model_set_current(channel, weight_index)

        # reset tournament counter and add matches
        n_players = len(weight_indices)
        self._training_db.tournament_reset()
        channels = list(range(1, 1 + n_players))
        for _ in range(n_rounds):
            for channel_1 in channels:
                for channel_2 in [c for c in channels if c != channel_1]:
                    self._training_db.tournament_add_match(channel_1, channel_2)
        n_expected_games = n_players * (n_players - 1) * n_rounds
        self._allocated_channels = channels
        return n_expected_games

    def _await_tournament_results(self, expected_games: int, deallocate_channels: bool = True) -> TournamentResult:
        with tqdm(desc="awaiting tournament", total=expected_games) as pbar:
            last_completed_games = 0
            while (completed_games := self._training_db.tournament_get_n_completed_games()) < expected_games:
                pbar.update(completed_games - last_completed_games)
                last_completed_games = completed_games
                time.sleep(1)

        # deallocate pred-channels
        if self._allocated_channels is not None and deallocate_channels:
            for channel in self._allocated_channels:
                self._training_db.model_remove(channel)
        return self._training_db.tournament_get_results()
