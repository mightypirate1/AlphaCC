import logging

import dill
import redis

from alpha_cc.agents.mcts.training_data import TrainingData
from alpha_cc.db.models.tournament_results import TournamentResult
from alpha_cc.db.redis_dbs import RedisDBs

POLL_TIMEOUT_SEC = 2


logger = logging.getLogger(__file__)


class TrainingDB:
    """
    Redis database for training data.

    Data categories:
    - training_data
    - model
    - weights
    - tournament
    """

    def __init__(
        self,
        host: str = "localhost",
    ) -> None:
        self._db = redis.Redis(host=host, db=RedisDBs.TRAINING.value)

    @property
    def queue_key(self) -> str:
        return "experience-queue"

    @property
    def latest_weights_key(self) -> str:
        return "weights-latest"

    @property
    def current_models_key(self) -> str:
        return "update-models"

    @property
    def latest_weights_index_key(self) -> str:
        return "latest-weights-index"

    @property
    def tournament_counter_key(self) -> str:
        return "tournament-counter"

    @property
    def tournament_results_key(self) -> str:
        return "tournament-results"

    @property
    def tournament_queue_key(self) -> str:
        return "tournament-queue"

    def weight_key(self, index: int | str, batch_size: int | None = None) -> str:
        base = f"weights-{str(index).zfill(4)}"
        if batch_size is not None:
            return f"{base}-b{batch_size}"
        return base

    def tournament_result_key(self, channel_1: int, channel_2: int, winner: int) -> str:
        return f"{channel_1}-{channel_2}-{winner}"

    def flush_db(self) -> None:
        logger.debug("reseting db")
        self._db.flushdb()

    ##
    # training_data
    def training_data_post(self, training_data: TrainingData) -> None:
        self._db.lpush(self.queue_key, dill.dumps(training_data))

    def training_data_fetch_all(self) -> list[TrainingData]:
        training_datas = []
        while training_data := self.training_data_fetch(blocking=False):
            training_datas.append(training_data)
        return training_datas

    def training_data_fetch(self, blocking: bool = False) -> TrainingData:
        def blocking_fetch() -> bytes | None:
            data: bytes | None = None
            while True:
                resp = self._db.brpop(self.queue_key, timeout=POLL_TIMEOUT_SEC)
                if resp is not None:
                    _, data = resp  # type: ignore
                    return data

        encoded_training_data = blocking_fetch() if blocking else self._db.rpop(self.queue_key)
        if encoded_training_data is None:
            return TrainingData(trajectory=[], internal_nodes={})
        training_data = dill.loads(encoded_training_data)  # noqa
        return training_data

    ##
    # model
    def model_set_current(self, channel: int, weight_index: int) -> None:
        self._db.hset(self.current_models_key, key=str(channel), value=str(weight_index))

    def model_remove(self, channel: int) -> None:
        self._db.hdel(self.current_models_key, str(channel))

    def model_get_current(self) -> dict[int, int]:
        channels = self._db.hkeys(self.current_models_key)
        if channels is None:
            return {}
        return {int(channel): int(self._db.hget(self.current_models_key, channel)) for channel in channels}  # type: ignore

    ##
    # weights
    def weights_incr_weights_index(self) -> int:
        return int(self._db.incr(self.latest_weights_index_key))  # type: ignore

    def weights_publish(self, payload: bytes, index: int, batch_size: int | None = None, set_latest: bool = False) -> None:
        self._db.set(self.weight_key(index, batch_size), payload)
        if set_latest:
            self._db.set(self.latest_weights_index_key, index)
            self._db.set(self.latest_weights_key, payload)
        logger.debug(f"published weights {index} (batch_size={batch_size})")

    def weights_fetch(self, index: int | str, batch_size: int | None = None) -> bytes:
        logger.debug(f"fetching weights {index} (batch_size={batch_size})")
        return self._db.get(self.weight_key(index, batch_size))  # type: ignore

    def weights_exists(self, index: int | str, batch_size: int | None = None) -> bool:
        return self._db.exists(self.weight_key(index, batch_size)) > 0

    def weights_fetch_latest_index(self) -> int:
        response = self._db.get(self.latest_weights_index_key)
        index = 0 if response is None else int(response)  # type: ignore
        logger.debug(f"latest index: {index}")
        return index

    ##
    # tournament
    def tournament_reset(self) -> None:
        self._db.set(self.tournament_counter_key, 0)
        self._db.delete(self.tournament_queue_key)
        self._db.delete(self.tournament_results_key)

    def tournament_increment_counter(self) -> None:
        self._db.incr(self.tournament_counter_key, 1)

    def tournament_add_match(self, channel_1: int, channel_2: int) -> None:
        paring = (channel_1, channel_2)
        self._db.lpush(self.tournament_queue_key, dill.dumps(paring))

    def tournament_get_match(self) -> tuple[int, int] | None:
        encoded_paring = self._db.rpop(self.tournament_queue_key)
        if encoded_paring is None:
            return None
        return dill.loads(encoded_paring)  # noqa

    def tournament_get_n_completed_games(self) -> int:
        return int(self._db.get(self.tournament_counter_key))  # type: ignore

    def tournament_add_result(self, channel_1: int, channel_2: int, winner: int) -> None:
        pairing_key = self.tournament_result_key(channel_1, channel_2, winner)
        self._db.hincrby(self.tournament_results_key, pairing_key, 1)

    def tournament_get_results(self) -> TournamentResult:
        return TournamentResult.from_db(self.tournament_results_key, self._db)
