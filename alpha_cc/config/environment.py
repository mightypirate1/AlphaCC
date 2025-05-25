import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

DOTENV_FILE = Path(__file__).parents[2] / ".env"
load_dotenv(DOTENV_FILE)


@dataclass
class Environment:
    redis_host_main: ClassVar[str] = os.environ.get("REDIS_HOST_MAIN", "localhost")
    redis_host_pred_1: ClassVar[str] = os.environ.get("REDIS_HOST_PRED_1", "localhost")
    redis_host_pred_2: ClassVar[str] = os.environ.get("REDIS_HOST_PRED_2", "localhost")
    redis_host_pred_3: ClassVar[str] = os.environ.get("REDIS_HOST_PRED_3", "localhost")
    redis_host_pred_4: ClassVar[str] = os.environ.get("REDIS_HOST_PRED_4", "localhost")
    tb_logdir: ClassVar[str] = os.environ.get("TB_LOGDIR", "data/logdir")
    model_dir: ClassVar[str] = os.environ.get("MODELDIR", "data/models/api")
