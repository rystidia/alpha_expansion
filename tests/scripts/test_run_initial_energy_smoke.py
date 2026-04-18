import os, sys, subprocess, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def test_runs_and_writes_csv():
    with tempfile.TemporaryDirectory() as out:
        env = os.environ | {"AE_RESULTS_DIR": out}
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "run_initial_energy.py"),
             "--instances", "chain", "--size", "10", "--reps", "2",
             "--inits", "zero,random", "--strategies", "sequential",
             "--no-trajectory"],
            check=True, env=env,
        )
        with open(os.path.join(out, "artificial.csv")) as f:
            lines = f.read().strip().splitlines()
        assert lines[0].startswith("instance,size,init,strategy,seed,")
        assert len(lines) >= 1 + 1 * 2 * 1 * 2
