import contextlib
import io
import logging
import threading
import warnings
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from alpha_cc.db import TrainingDB
from alpha_cc.nn.nets.default_net import DefaultNet
from alpha_cc.training import model_io

logger = logging.getLogger(__name__)

for _name in ("torch.onnx", "onnxscript", "onnx", "onnx_ir"):
    logging.getLogger(_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module=r"torch\.onnx")
warnings.filterwarnings("ignore", message=r"isinstance\(treespec, LeafSpec\)")


class ExportThread(threading.Thread):
    """
    Background thread that handles ONNX export, Redis publish, and disk save.

    Uses a "latest slot" pattern: the main thread always overwrites the pending
    slot, so if the export thread falls behind it skips intermediate weights and
    picks up the newest. Weight index gaps are fine — the nn-service handles them.

    Crucially, model_set_current is called here (after weights exist in Redis),
    not in the main thread, to avoid a race where the nn-service is told about
    index N before weights-N has been published.
    """

    def __init__(
        self,
        model: DefaultNet,
        db: TrainingDB,
        run_id: str,
        board_size: int,
        onnx_compiled_batch_size: int | None = None,
        summary_writer: SummaryWriter | None = None,
    ) -> None:
        super().__init__(daemon=True, name="export-thread")
        self._model = model
        self._db = db
        self._run_id = run_id
        self._board_size = board_size
        self._onnx_compiled_batch_size = onnx_compiled_batch_size
        self._summary_writer = summary_writer

        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._pending: tuple[dict[str, Any], int] | None = None
        self._processing = False
        self._work_available = threading.Event()

    def submit(self, state_dict: dict[str, Any], curr_index: int) -> None:
        with self._lock:
            if self._pending is not None:
                logger.warning(f"export-thread: dropped weights {self._pending[1]} (export lagging behind training)")
            self._pending = (state_dict, curr_index)
        self._work_available.set()

    def wait_idle(self) -> None:
        """Block until no work is pending and the thread is not currently exporting."""
        with self._idle:
            self._idle.wait_for(lambda: not self._processing and self._pending is None)

    def run(self) -> None:
        while True:
            self._work_available.wait()
            with self._lock:
                if self._pending is None:
                    self._work_available.clear()
                    continue
                state_dict, curr_index = self._pending
                self._pending = None
                self._processing = True
                self._work_available.clear()

            try:
                self._model.load_state_dict(state_dict)
                onnx_payload = self._serialize()
                self._db.weights_publish(
                    onnx_payload,
                    curr_index,
                    batch_size=self._onnx_compiled_batch_size,
                    set_latest=True,
                )
                self._db.model_set_current(0, curr_index)
                model_io.save_weights(self._run_id, curr_index, state_dict, onnx_payload)
                logger.info(f"export-thread: published weights {curr_index}")
                if self._summary_writer is not None:
                    self._summary_writer.add_scalar("trainer/export-index", curr_index, global_step=curr_index)
            except Exception:
                logger.exception("export-thread error")
            finally:
                with self._idle:
                    self._processing = False
                    self._idle.notify_all()

    def _serialize(self) -> bytes:
        from pathlib import Path

        from alpha_cc.config import Environment

        self._model.eval()
        batch = self._onnx_compiled_batch_size or 1
        dummy = torch.zeros(batch, 2, self._board_size, self._board_size)
        tmp_path = Path(Environment.model_dir) / "temp-export-thread.onnx"
        dynamic_axes = (
            None
            if self._onnx_compiled_batch_size is not None
            else {
                "input": {0: "batch"},
                "policy": {0: "batch"},
                "value": {0: "batch"},
            }
        )
        with contextlib.redirect_stdout(io.StringIO()):
            torch.onnx.export(
                self._model,
                (dummy,),
                tmp_path,
                input_names=["input"],
                output_names=["policy", "value"],
                dynamic_axes=dynamic_axes,
                opset_version=18,
                do_constant_folding=True,
                external_data=False,
                verbose=False,
            )
        with open(tmp_path, "rb") as f:
            payload = f.read()
        self._model.train()
        return payload
