import logging
import threading

from torch.utils.tensorboard import SummaryWriter

from alpha_cc.db import TrainingDB
from alpha_cc.db.models import TournamentResult
from alpha_cc.runtimes import TournamentRuntime

logger = logging.getLogger(__file__)


class TournamentManager:
    def __init__(
        self,
        run_id: str,
        champion_index: int,
        tournament_runtime: TournamentRuntime,
        training_db: TrainingDB,
        summary_writer: SummaryWriter,
    ) -> None:
        self._run_id = run_id
        self._champion_index = champion_index
        self._tournament_runtime = tournament_runtime
        self._training_db = training_db
        self._summary_writer = summary_writer
        self._tournament_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._tournament_thread is not None and self._tournament_thread.is_alive()

    @property
    def champion_index(self) -> int:
        return self._champion_index

    def run_tournament(self, challenger_idx: int) -> None:
        """
        Arranges tournament, waits for the workers to play it out,
        and records the results to the StatsWriter.
        """

        def _run_tournament() -> None:
            tournament_results = self._tournament_runtime.run_tournament(
                challenger_idx, self._champion_index, n_rounds=5
            )
            win_rate, win_rate_as_white, win_rate_as_black = self._extract_winrate(tournament_results)
            if win_rate > 0.55:
                self._champion_index = challenger_idx
                logger.info(f"new champion: {self._champion_index}! (winrate={win_rate})")
            self._log_winrate(win_rate, win_rate_as_white, win_rate_as_black, challenger_idx)

        if self.is_running:
            logger.warning("tournament already running; it will be terminated")
            self._training_db.tournament_reset()
            self._tournament_thread.join(timeout=5.0)  # type: ignore[union-attr]
            if self._tournament_thread and self._tournament_thread.is_alive():
                logger.warning("Tournament thread did not terminate within timeout")

        self._tournament_thread = threading.Thread(target=_run_tournament, daemon=True)
        self._tournament_thread.start()

    def _extract_winrate(self, tournament_results: TournamentResult) -> tuple[float, float, float]:
        # assumes exactly two players were queued to play the tournament:
        #  1. current weights
        #  2. champion weights
        win_rate_as_white = tournament_results[1, 2]
        win_rate_as_black = 1 - tournament_results[2, 1]
        win_rate = (win_rate_as_white + win_rate_as_black) / 2
        return win_rate, win_rate_as_white, win_rate_as_black

    def _log_winrate(
        self,
        win_rate: float,
        win_rate_as_white: float,
        win_rate_as_black: float,
        challenger_idx: int,
    ) -> float:
        step = challenger_idx
        logger.info(f"WINRATES: total={win_rate} as_white={win_rate_as_white}, as_black={win_rate_as_black}")
        self._summary_writer.add_scalar("tournament/win-rate", win_rate, step)
        self._summary_writer.add_scalar("tournament/win-rate-as-white", win_rate_as_white, step)
        self._summary_writer.add_scalar("tournament/win-rate-as-black", win_rate_as_black, step)
        self._summary_writer.add_scalar("tournament/champion-index", self._champion_index, step)
        return win_rate
