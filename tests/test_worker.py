import sys
import subprocess
import textwrap
from pathlib import Path

def test_worker_doesnt_load_torch() -> None:
    code = textwrap.dedent("""
        import sys
        from alpha_cc.entrypoints.worker_thread import main  # triggers its import chain
        assert 'torch' not in sys.modules, 'Torch should not be loaded by worker_thread entrypoint'
    """)
    proc = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
