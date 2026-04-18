import os, sys, subprocess, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def test_runs_and_writes_csv():
    with tempfile.TemporaryDirectory() as out:
        env = os.environ | {"AE_RESULTS_DIR": out}
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "run_worst_case.py"),
             "--sizes", "5,7", "--strategies", "sequential", "--solvers", "bk"],
            check=True, env=env, capture_output=True, text=True,
        )
        csv_path = os.path.join(out, "sweep.csv")
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            lines = f.read().strip().splitlines()
        assert lines[0].startswith("instance,size,")
        assert len(lines) >= 1 + 3 * 2
