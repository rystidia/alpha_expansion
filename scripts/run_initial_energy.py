import argparse
import csv
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiments import (
    build_chain, build_checkerboard, build_snake,
    build_restoration_model, build_stereo_model,
    init_zero, init_random, init_partial_optimum, run_one, make_strategy,
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
import alpha_expansion_py as ae

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RESULTS = os.path.join(ROOT, "data", "results", "initial_energy")
DEFAULT_PLOTS = os.path.join(ROOT, "data", "plots", "initial_energy")

INSTANCE_BUILDERS = {
    "chain": build_chain,
    "checkerboard": build_checkerboard,
    "snake": build_snake,
}


def make_init(init_name, model, optimum, seed):
    if init_name == "zero":
        return init_zero(model)
    if init_name == "random":
        return init_random(model, seed=seed)
    if init_name.startswith("partial_"):
        frac = int(init_name.split("_")[1]) / 100.0
        return init_partial_optimum(model, optimum, frac, seed=seed)
    raise ValueError(init_name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["artificial", "real"], default="artificial")
    p.add_argument("--instances", default="chain,checkerboard,snake")
    p.add_argument("--datasets", default="tsukuba,cones")
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--inits", default="zero,random,partial_10,partial_50")
    p.add_argument("--strategies", default="sequential,greedy,randomized")
    p.add_argument("--solver", default="bk")
    p.add_argument("--max-cycles", type=int, default=2000)
    p.add_argument("--no-trajectory", action="store_true")
    return p.parse_args()


def _build_real_model(name: str):
    if name == "tsukuba":
        from ci_data import load_tsukuba
        left, right, _ = load_tsukuba()
        return build_stereo_model(left, right, num_labels=16)
    if name == "cones":
        from datasets import load_middlebury_stereo
        left, right, gt, scale = load_middlebury_stereo("cones")
        num_labels = int(gt.max() / scale) + 1
        return build_stereo_model(left, right, num_labels=num_labels)
    if name in ("airplane", "house", "peppers"):
        from datasets import load_restoration_image, add_noise
        clean = load_restoration_image(name)
        noisy = add_noise(clean, sigma=20.0)
        model, _ = build_restoration_model(noisy, num_labels=16, lambda_smooth=750)
        return model
    raise ValueError(f"unknown real dataset: {name}")


def _proxy_optimum(model, solver: str, max_cycles: int) -> list:
    labels = init_zero(model)
    model.set_labels(labels)
    run_one(model, "sequential", solver, max_cycles)
    return model.get_labels()


def main():
    args = parse_args()
    out_dir = os.environ.get("AE_RESULTS_DIR", DEFAULT_RESULTS)
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "real":
        _main_real(args, out_dir)
        return

    rows = []
    for instance in args.instances.split(","):
        builder = INSTANCE_BUILDERS[instance]
        for init in args.inits.split(","):
            for strategy in args.strategies.split(","):
                for seed in range(args.reps):
                    model, optimum = builder(args.size)
                    labels = make_init(init, model, optimum, seed)
                    res = run_one(model, strategy, args.solver,
                                  args.max_cycles, init_labels=labels)
                    res.update(instance=instance, size=args.size, init=init,
                               strategy=strategy, seed=seed)
                    rows.append(res)

    csv_path = os.path.join(out_dir, "artificial.csv")
    cols = ["instance", "size", "init", "strategy", "seed",
            "cycles", "moves_attempted", "initial_energy", "final_energy", "wall_seconds"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"wrote {csv_path}")

    if not args.no_trajectory:
        _scatter_plots(rows, key="instance", out_dir=DEFAULT_PLOTS)
        _trajectory_plots(args)


def _main_real(args, out_dir):
    rows = []
    cols = ["dataset", "init", "strategy", "seed",
            "cycles", "moves_attempted", "initial_energy", "final_energy", "wall_seconds"]
    for name in args.datasets.split(","):
        proxy = _proxy_optimum(_build_real_model(name), args.solver, args.max_cycles)
        for init in args.inits.split(","):
            for strategy in args.strategies.split(","):
                for seed in range(args.reps):
                    model = _build_real_model(name)
                    if init == "zero":
                        labels = init_zero(model)
                    elif init == "random":
                        labels = init_random(model, seed=seed)
                    elif init.startswith("partial_"):
                        frac = int(init.split("_")[1]) / 100.0
                        labels = init_partial_optimum(model, proxy, frac, seed=seed)
                    else:
                        raise ValueError(init)
                    res = run_one(model, strategy, args.solver, args.max_cycles,
                                  init_labels=labels)
                    res.update(dataset=name, init=init, strategy=strategy, seed=seed)
                    rows.append(res)
                    print(f"{name:12} {init:12} {strategy:10} seed={seed} "
                          f"cycles={res['cycles']} energy={res['final_energy']}")

    csv_path = os.path.join(out_dir, "real.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"wrote {csv_path}")

    if not args.no_trajectory:
        _scatter_plots(rows, key="dataset", out_dir=os.path.join(DEFAULT_PLOTS, "real"))


def _scatter_plots(rows, key, out_dir):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for group in {r[key] for r in rows}:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for init in sorted({r["init"] for r in rows if r[key] == group}):
            xs = [r["initial_energy"] for r in rows
                  if r[key] == group and r["init"] == init]
            ys = [r["moves_attempted"] for r in rows
                  if r[key] == group and r["init"] == init]
            ax.scatter(xs, ys, label=init, alpha=0.7)
        ax.set_xlabel("initial energy")
        ax.set_ylabel("moves attempted")
        ax.set_title(f"{group}: moves attempted vs initial energy")
        ax.legend()
        ax.grid(alpha=0.3)
        out = os.path.join(out_dir, f"{group}_scatter.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")


def _alpha_sequence(strategy, num_labels, max_cycles, seed):
    if strategy == "sequential":
        for _ in range(max_cycles):
            for a in range(num_labels):
                yield a
    elif strategy == "randomized":
        rng = np.random.default_rng(seed)
        for _ in range(max_cycles):
            for a in rng.permutation(num_labels):
                yield int(a)
    else:
        raise ValueError(f"trajectory plot does not support strategy {strategy}")


def _trajectory_plots(args):
    import matplotlib.pyplot as plt
    os.makedirs(DEFAULT_PLOTS, exist_ok=True)
    supported = [s for s in args.strategies.split(",") if s in {"sequential", "randomized"}]
    if not supported:
        return
    for instance in args.instances.split(","):
        builder = INSTANCE_BUILDERS[instance]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for init in args.inits.split(","):
            for strategy in supported:
                model, optimum = builder(args.size)
                labels = make_init(init, model, optimum, seed=0)
                model.set_labels(labels)
                optimizer = ae.AlphaExpansionInt(model, args.solver)
                energies = [model.evaluate_total_energy()]
                step = 0
                for alpha in _alpha_sequence(strategy, model.num_labels, args.max_cycles, seed=0):
                    optimizer.perform_expansion_move(alpha)
                    energies.append(model.evaluate_total_energy())
                    step += 1
                    if step % model.num_labels == 0:
                        if energies[-1] == energies[-1 - model.num_labels]:
                            break
                ax.plot(energies, label=f"{init} / {strategy}")
        ax.set_xlabel("alpha-move")
        ax.set_ylabel("total energy")
        ax.set_title(f"{instance}: energy trajectory")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        out = os.path.join(DEFAULT_PLOTS, f"{instance}_trajectory.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
