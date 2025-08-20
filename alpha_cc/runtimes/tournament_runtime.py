import time
from datetime import datetime
from logging import getLogger

import numpy as np
from tqdm_loggable.auto import tqdm

from alpha_cc.agents.mcts.mcts_agent import MCTSAgent
from alpha_cc.db import GamesDB, TrainingDB
from alpha_cc.db.models import TournamentResult
from alpha_cc.engine import Board

logger = getLogger(__file__)


class TournamentRuntime:
    """
    Class for arranging and runnining tournaments.
    - expects to be run across multiple services
    - currently supports only 2-player tournaments

    TODO: this class should be refactored/rewritten from scratch,
    and be merged with TournamentManager in the trainer theread endpoint.
    """

    def __init__(
        self, size: int, training_db: TrainingDB, games_db: GamesDB | None = None, max_game_length: int | None = None
    ) -> None:
        self._size = size
        self._training_db = training_db
        self._games_db = games_db
        self._max_game_duration = np.inf if max_game_length is None else max_game_length
        self._allocated_channels: list[int] | None = None  # used to help creator of tournament to clean up

    def play_and_record_game(self, agent_channel_dict: dict[int, MCTSAgent]) -> None:
        game_id = f"tournament-game/{datetime.now().strftime(r'%Y-%m-%d-%H-%M-%S')}"
        player_1, player_2 = agent_channel_dict.keys()
        if self._games_db is not None:
            self._games_db.create_game(game_id, self._size)
        # since trainer will block until tournament is over,
        # we try-catch and post back the result in case of failure.
        # there is still a rare bug causing a crash
        try:
            board = Board(self._size)
            agents = tuple(agent_channel_dict.values())
            current_agent_idx = 0

            actions = []
            with tqdm(desc="tournament-game", total=self._max_game_duration) as pbar:
                while not board.info.game_over and board.info.duration < self._max_game_duration:
                    # get and record action/board
                    agent = agents[current_agent_idx]
                    action_index = agent.choose_move(board, training=False)
                    actions.append(action_index)
                    # apply action
                    move = board.get_moves()[action_index]
                    board = board.apply(move)
                    pbar.update(1)
                    current_agent_idx = 1 - current_agent_idx
                    if self._games_db is not None:
                        # record action
                        self._games_db.add_move(game_id, action_index)

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

    def run_tournament(self, challenger_idx: int, champion_idx: int, n_rounds: int = 5) -> TournamentResult:
        """
        Arranges tournament by setting up the db accordingly and waiting for the results.
        """
        expected_games = self._arrange_tournament(challenger_idx, champion_idx, n_rounds)
        return self._await_tournament_results(expected_games)

    def _arrange_tournament(self, challenger_idx: int, champion_idx: int, n_rounds: int = 5) -> int:
        """
        Arranges a tournament by:
        - setting the model configuration to the given weight indices
        - adding matches to the tournament queue

        Returns the expected number of games

        """
        # assign weights to tournament channels
        n_players = 2
        channels = [1, 2]  # 0 is the current weights
        self._training_db.model_set_current(1, challenger_idx)
        self._training_db.model_set_current(2, champion_idx)

        # reset tournament counter and add matches
        self._training_db.tournament_reset()
        for _ in range(n_rounds):
            for channel_1 in channels:
                for channel_2 in [c for c in channels if c != channel_1]:
                    self._training_db.tournament_add_match(channel_1, channel_2)
        n_expected_games = n_players * (n_players - 1) * n_rounds
        self._allocated_channels = channels
        return n_expected_games

    def _await_tournament_results(self, expected_games: int, deallocate_channels: bool = True) -> TournamentResult:
        with tqdm(desc="tournament: awaiting", total=expected_games) as pbar:
            last_completed_games = 0
            while (completed_games := self._training_db.tournament_get_n_completed_games()) < expected_games:
                pbar.update(completed_games - last_completed_games)
                last_completed_games = completed_games
                time.sleep(1)
        logger.info("tournament: completed")

        # deallocate pred-channels
        if self._allocated_channels is not None and deallocate_channels:
            for channel in self._allocated_channels:
                logger.info(f"tournament: deallocating channel {channel}")
                self._training_db.model_remove(channel)
        logger.info("tournament: fetching results")
        return self._training_db.tournament_get_results()
