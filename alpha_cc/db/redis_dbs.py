from enum import Enum


class RedisDBs(Enum):
    """
    Enum for Redis databases.
    """

    TRAINING = 0  # for moving training data
    PRED_QUEUE = 1  # for workers to request predictions (used by rust)
    PRED_CACHE = 2  # for nn service to output predictions (used by rust)
    BACKEND = 3  # for backend to store data
