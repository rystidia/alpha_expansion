import argparse
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiments import build_chain, build_checkerboard, build_snake, run_one

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RESULTS = os.path.join(ROOT, "data", "results", "worst_case")
DEFAULT_PLOTS = os.path.join(ROOT, "data", "plots", "worst_case")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="10,20,40,80,160")
    p.add_argument("--strategies", default="sequential,greedy,randomized")
    p.add_argument("--solvers", default="bk")
    p.add_argument("--max-cycles", type=int, default=100000)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    strategies = args.strategies.split(",")
    solvers = args.solvers.split(",")

    out_dir = os.environ.get("AE_RESULTS_DIR", DEFAULT_RESULTS)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sweep.csv")

    BUILDERS = {"chain": build_chain, "checkerboard": build_checkerboard, "snake": build_snake}

    rows = []
    for size in sizes:
        for name, builder in BUILDERS.items():
            for strategy in strategies:
                for solver in solvers:
                    model, _opt = builder(size)
                    res = run_one(model, strategy, solver, args.max_cycles)
                    res.update(instance=name, size=size, strategy=strategy, solver=solver)
                    rows.append(res)
                    print(f"{name:13} n={size:4} {strategy:10} {solver:8} "
                          f"cycles={res['cycles']:4} E={res['final_energy']}")

    cols = ["instance", "size", "strategy", "solver",
            "cycles", "moves_applied", "initial_energy", "final_energy", "wall_seconds"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"wrote {csv_path}")

    if not args.no_plot:
        _plot(rows)


def _plot(rows):
    import numpy as np
    import matplotlib.pyplot as plt
    plot_dir = os.path.join(ROOT, "data", "plots", "worst_case")
    os.makedirs(plot_dir, exist_ok=True)

    # reference exponent and label per instance
    ref_exp = {"chain": 1.0, "checkerboard": 0.5, "snake": 1.0}
    ref_label = {"chain": "O(n)", "checkerboard": "O(√n)", "snake": "O(n)"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, instance in zip(axes, ["chain", "checkerboard", "snake"]):
        all_xs = []
        for strategy in sorted({r["strategy"] for r in rows}):
            xs, ys = [], []
            for r in rows:
                if r["instance"] == instance and r["strategy"] == strategy:
                    n = int(r["size"]) if instance == "chain" else int(r["size"]) ** 2
                    xs.append(n); ys.append(int(r["moves_applied"]))
            xs, ys = zip(*sorted(zip(xs, ys)))
            ax.loglog(xs, ys, "o-", label=strategy)
            all_xs.extend(xs)

        # reference line anchored to the midpoint of the data
        ns = np.array(sorted(set(all_xs)), dtype=float)
        mid = ns[len(ns) // 2]
        mid_y = np.median([int(r["moves_applied"]) for r in rows
                           if r["instance"] == instance and
                           (int(r["size"]) if instance == "chain" else int(r["size"])**2) == int(mid)])
        exp = ref_exp[instance]
        ref_ys = mid_y * (ns / mid) ** exp
        ax.loglog(ns, ref_ys, "--", color="gray", linewidth=1.2, label=ref_label[instance])

        ax.set_title(instance)
        ax.set_xlabel("n (nodes)")
        ax.set_ylabel("moves applied")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = os.path.join(plot_dir, "moves_vs_size.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
