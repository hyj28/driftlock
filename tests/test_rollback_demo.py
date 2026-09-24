from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_STDOUT = """\
driftlock rollback demo
scripted stand-in: no model is called; the stall is planted deliberately
no fine judge: heuristics-only mode rolls every coarse trigger back unreviewed
checkpoint: checkpoint-0 (initial, logical step 0, phase=new)
step: sequence=1 logical=1 attempt=1 action=write healthy progress 1
step: sequence=2 logical=2 attempt=1 action=write healthy progress 2
step: sequence=3 logical=3 attempt=1 action=write healthy progress 3
step: sequence=4 logical=4 attempt=1 action=write healthy progress 4
checkpoint: checkpoint-1 (accepted, logical step 4, phase=healthy-4)
step: sequence=5 logical=5 attempt=1 action=write healthy progress 5
step: sequence=6 logical=6 attempt=1 action=write healthy progress 6
step: sequence=7 logical=7 attempt=1 action=write healthy progress 7
step: sequence=8 logical=8 attempt=1 action=repeat stalled detour
checkpoint: checkpoint-2 (accepted, logical step 8, phase=stalled)
step: sequence=9 logical=9 attempt=1 action=repeat stalled detour
step: sequence=10 logical=10 attempt=1 action=repeat stalled detour
coarse trigger: action_loop (lookback 6 steps)
rollback: restored checkpoint-1 (accepted, logical step 4, phase=healthy-4)
retry observes: progress.txt='healthy 4', detour.txt exists=False
step: sequence=11 logical=5 attempt=2 action=finish from restored draft
checkpoint: checkpoint-3 (terminal, logical step 5, phase=finished)
RunStatus: completed
rollbacks: 1
"""


def test_rollback_demo_runs_at_program_seam_and_is_deterministic(
    tmp_path: Path,
) -> None:
    demo = Path(__file__).resolve().parents[1] / "examples" / "rollback_demo.py"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.endswith("_API_KEY") and not key.startswith("OPENROUTER_")
    }

    runs = [
        subprocess.run(
            [sys.executable, str(demo)],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for _ in range(2)
    ]

    for run in runs:
        assert run.returncode == 0, run.stderr
        assert run.stderr == ""
        assert run.stdout == EXPECTED_STDOUT
    assert runs[0].stdout == runs[1].stdout
    assert list(tmp_path.iterdir()) == []
    assert list(Path(tempfile.gettempdir()).glob("driftlock-rollback-demo-*")) == []
