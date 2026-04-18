import subprocess
import sys
import os
import tempfile

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


def test_run_initial_energy_real_tsukuba_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "run_initial_energy.py",
             "--mode", "real",
             "--datasets", "tsukuba",
             "--inits", "zero,random",
             "--strategies", "sequential",
             "--reps", "1",
             "--no-trajectory"],
            cwd=SCRIPTS,
            capture_output=True, text=True,
            env={**os.environ, "AE_RESULTS_DIR": tmp},
        )
        assert result.returncode == 0, result.stderr
        csv_path = os.path.join(tmp, "real.csv")
        assert os.path.exists(csv_path)
        lines = open(csv_path).readlines()
        assert lines[0].startswith("dataset,init,strategy,seed,")
        assert len(lines) >= 3
