from typing import Any

import dill
import redis

from alpha_cc.agents.mcts import MCTSExperience


class TrainingDB:
    def __init__(self, host: str = "localhost") -> None:
        self._db = redis.Redis(host=host)

    @property
    def queue_key(self) -> str:
        return "experience-queue"

    @property
    def latest_weights_key(self) -> str:
        return "weights-latest"

    @property
    def latest_weights_index_key(self) -> str:
        return "latest-weights-index"

    def weight_key(self, index: int | str) -> str:
        return f"weights-{str(index).zfill(4)}"

    def post_experiences(self, experiences: list[MCTSExperience]) -> None:
        self._db.lpush(self.queue_key, dill.dumps(experiences))

    def fetch_experiences(self, blocking: bool = False) -> list[MCTSExperience]:
        if blocking:
            _, encoded_experiences = self._db.brpop(self.queue_key, timeout=0)  # type: ignore
        else:
            encoded_experiences = self._db.rpop(self.queue_key)
            if encoded_experiences is None:
                return []
        return dill.loads(encoded_experiences)  # noqa

    def publish_latest_weights(self, state_dict: dict[str, Any]) -> int:
        payload = dill.dumps(state_dict)
        self._db.set(self.latest_weights_key, payload)
        current_index = self._db.incr(self.latest_weights_index_key)
        self._db.set(self.weight_key(str(current_index)), payload)
        return int(current_index)  # type: ignore

    def fetch_latest_weights_with_index(self) -> tuple[int, dict[str, Any]]:
        return self.fetch_latest_weight_index(), self.fetch_latest_weights()

    def fetch_latest_weights(self) -> dict[str, Any]:
        encoded_weights = self._db.get(self.latest_weights_key)
        return dill.loads(encoded_weights)  # noqa

    def fetch_weights(self, index: int | str) -> dict[str, Any]:
        encoded_weights = self._db.get(self.weight_key(index))
        return dill.loads(encoded_weights)  # noqa

    def fetch_latest_weight_index(self) -> int:
        response = self._db.get(self.latest_weights_index_key)
        return 0 if response is None else int(response)  # type: ignore

    def weights_is_latest(self, current_index: int) -> bool:
        latest_index = self.fetch_latest_weight_index()
        return latest_index > current_index

    def first_weights_published(self) -> bool:
        response = self._db.get(self.latest_weights_index_key)
        return response is not None
