from typing import Any


class MockFileWriter:
    """
    Mock class that enables testing (parts of) SummaryWriter without writing to disk.
    """

    def __init__(self, log_dir: str, max_queue: int = 10, flush_secs: int = 120, filename_suffix: str = "") -> None:
        pass

    def get_logdir(self) -> None:
        pass

    def add_event(self, event: Any, step: Any = None, walltime: Any = None) -> None:
        pass

    def add_summary(self, summary: Any, global_step: Any = None, walltime: Any = None) -> None:
        pass

    def add_graph(self, graph_profile: Any, walltime: Any = None) -> None:
        pass

    def add_onnx_graph(self, graph: Any, walltime: Any = None) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def reopen(self) -> None:
        pass
