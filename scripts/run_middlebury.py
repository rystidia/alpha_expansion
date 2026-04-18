import sys
import os
import csv
import numpy as np
import argparse
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))

import alpha_expansion_py as ae
from ci_data import load_tsukuba
from experiments import build_stereo_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["bk", "ortools"], default="bk")
    parser.add_argument("--strategy", choices=["sequential", "greedy", "randomized"],
                        default="sequential")
    parser.add_argument("--max_cycles", type=int, default=100)
    parser.add_argument("--scene", default="tsukuba",
                        help="Stereo scene: tsukuba (default), venus, cones")
    parser.add_argument("--output-csv", default=None,
                        help="Optional path to write results CSV")
    args = parser.parse_args()

    if args.scene == "tsukuba":
        left, right, gt = load_tsukuba()
        scale = 16
        num_labels = 16
    else:
        from datasets import load_middlebury_stereo
        left, right, gt, scale = load_middlebury_stereo(args.scene)
        num_labels = int(gt.max() / scale) + 1

    height, width = left.shape[:2]
    print(f"Scene: {args.scene}  size: {width}x{height}  labels: {num_labels}")

    model = build_stereo_model(left, right, num_labels)

    print(f"Initial energy: {model.evaluate_total_energy()}")
    print(f"Running Alpha Expansion ({args.solver}, {args.strategy})")

    opt = ae.AlphaExpansionInt(model, args.solver)
    if args.strategy == "sequential":
        strategy = ae.SequentialStrategyInt(args.max_cycles)
    elif args.strategy == "greedy":
        strategy = ae.GreedyStrategyInt(args.max_cycles)
    else:
        strategy = ae.RandomizedStrategyInt(args.max_cycles)

    cycles = strategy.execute(opt, model)
    final_energy = model.evaluate_total_energy()
    print(f"Converged in {cycles} cycles.  Final energy: {final_energy}")

    max_label = num_labels - 1
    gt_disparities = np.clip(gt / scale, 0, max_label).astype(np.int32)
    gt_energy = model.evaluate_total_energy_with_labels(gt_disparities.flatten().tolist())
    print(f"Ground truth energy: {gt_energy}")

    if args.scene == "tsukuba":
        if final_energy > gt_energy:
            print(f"FAIL: final energy ({final_energy}) is worse than ground truth ({gt_energy})")
            sys.exit(1)
        print(f"OK: final energy ({final_energy}) <= ground truth energy ({gt_energy})")

    labels = np.array(model.get_labels(), dtype=np.uint8).reshape((height, width))
    out_path = os.path.join(ROOT, "data", args.scene, "output.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(labels * scale).save(out_path)
    print(f"Saved disparity map to {out_path}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["scene", "solver", "strategy",
                                               "cycles", "final_energy", "gt_energy"])
            w.writeheader()
            w.writerow(dict(scene=args.scene, solver=args.solver,
                            strategy=args.strategy, cycles=cycles,
                            final_energy=final_energy, gt_energy=gt_energy))
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
