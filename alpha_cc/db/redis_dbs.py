from enum import Enum


class RedisDBs(Enum):
    """
    Enum for Redis databases.
    """

    TRAINING = 0  # for moving training data
    BACKEND = 1  # for backend to store data
    # higher numbers are reserved for PredDBChannel predictions
