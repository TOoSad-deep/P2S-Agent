import subprocess
import sys
from pathlib import Path


def test_cli_imports_without_fastapi():
    """p2s_agent.cli must import and expose main() without pulling in fastapi.

    Run in a fresh subprocess so we never mutate this interpreter's sys.modules
    (deleting fastapi here would corrupt later tests that rely on its class identity).
    """
    backend = Path(__file__).resolve().parents[2]
    code = (
        "import sys, p2s_agent.cli as c; "
        "assert hasattr(c, 'main'), 'cli must expose main()'; "
        "assert 'fastapi' not in sys.modules, 'importing p2s_agent.cli must not import fastapi'; "
        "print('CLI_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(backend),
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"cli import check failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert "CLI_OK" in proc.stdout
