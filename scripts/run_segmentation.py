import argparse, csv, os, sys
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import load_segmentation_image
from experiments import run_one
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
import alpha_expansion_py as ae

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def make_seeds(rgb_shape, bbox, bg_margin=30, fg_inset=30):
    h, w = rgb_shape[:2]
    x, y, bw, bh = bbox
    seeds = np.full((h, w), -1, dtype=np.int32)
    seeds[:max(0, y - bg_margin), :] = 0
    seeds[min(h, y + bh + bg_margin):, :] = 0
    seeds[:, :max(0, x - bg_margin)] = 0
    seeds[:, min(w, x + bw + bg_margin):] = 0
    seeds[y + fg_inset: y + bh - fg_inset, x + fg_inset: x + bw - fg_inset] = 1
    return seeds


def build_model(rgb, seeds, lambda_smooth=50.0):
    h, w, _ = rgb.shape
    data = rgb.astype(np.float64)
    num_labels = 2
    means, cov_invs = [], []
    for label in range(num_labels):
        pixels = data[seeds == label].reshape(-1, 3)
        if len(pixels) == 0:
            raise RuntimeError(f"label {label} has no seed pixels")
        means.append(pixels.mean(axis=0))
        cov = np.cov(pixels.T) + np.eye(3) * 1e-6
        cov_invs.append(np.linalg.inv(cov))
    flat = data.reshape(-1, 3)
    unary = np.zeros((h * w, num_labels), dtype=np.float64)
    for label in range(num_labels):
        diff = flat - means[label]
        mahal = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_invs[label], diff))
        unary[:, label] = mahal * 1000
    MAX = 10000.0
    seeds_flat = seeds.flatten()
    for label in range(num_labels):
        unary[seeds_flat == label, label] = 0.0
        for other in range(num_labels):
            if other != label:
                unary[seeds_flat == other, label] = MAX
    model = ae.EnergyModel(h * w, num_labels, "int32")
    model.set_unary_costs(unary.flatten().astype(np.int32).tolist())
    sigma_sq = float(np.var(data))
    node_ids = np.arange(h * w).reshape(h, w)
    h_n1 = node_ids[:, :-1].flatten()
    h_n2 = node_ids[:, 1:].flatten()
    h_diff = np.sum((data[:, :-1] - data[:, 1:]) ** 2, axis=2).flatten()
    h_w = (lambda_smooth * np.exp(-h_diff / (2 * sigma_sq))).astype(np.int32)
    v_n1 = node_ids[:-1, :].flatten()
    v_n2 = node_ids[1:, :].flatten()
    v_diff = np.sum((data[:-1, :] - data[1:, :]) ** 2, axis=2).flatten()
    v_w = (lambda_smooth * np.exp(-v_diff / (2 * sigma_sq))).astype(np.int32)
    n1 = np.concatenate([h_n1, v_n1]).tolist()
    n2 = np.concatenate([h_n2, v_n2]).tolist()
    weights = np.concatenate([h_w, v_w]).tolist()
    model.add_grid_edges(w, h)
    model.set_edge_weights(n1, n2, weights)
    return model


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt > 127
    inter = float(np.logical_and(p, g).sum())
    union = float(np.logical_or(p, g).sum())
    return inter / union if union > 0 else 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="flower,teddy,llama")
    p.add_argument("--strategies", default="sequential,greedy,randomized")
    p.add_argument("--solvers", default="bk,ortools")
    p.add_argument("--max-cycles", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir  = os.path.join(ROOT, "data", "results", "segmentation")
    plot_dir = os.path.join(ROOT, "data", "plots", "segmentation")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    rows = []
    comparison_data = []
    for name in args.images.split(","):
        rgb, gt, bbox = load_segmentation_image(name)
        seeds = make_seeds(rgb.shape, bbox, bg_margin=0, fg_inset=85)
        best_mask, best_iou = None, -1.0
        for strategy in args.strategies.split(","):
            for solver in args.solvers.split(","):
                model = build_model(rgb, seeds)
                res = run_one(model, strategy, solver, args.max_cycles)
                labels = np.array(model.get_labels()).reshape(rgb.shape[:2])
                mask = (labels * 255).astype(np.uint8)
                Image.fromarray(mask).save(
                    os.path.join(plot_dir, f"{name}_{strategy}_{solver}.png"))
                score = iou(labels, gt)
                if score > best_iou:
                    best_iou, best_mask = score, mask
                res.update(image=name, strategy=strategy, solver=solver, iou=score)
                rows.append(res)
                print(f"{name:8} {strategy:10} {solver:8} "
                      f"cycles={res['cycles']:3} IoU={score:.3f}")
        comparison_data.append((name, rgb, gt, best_mask, best_iou, bbox))

    cols = ["image", "strategy", "solver", "cycles", "moves_attempted",
            "initial_energy", "final_energy", "wall_seconds", "iou"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    _comparison_plot(comparison_data, plot_dir)


def _comparison_plot(comparison_data, plot_dir):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    n = len(comparison_data)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]
    for ax_row, (name, rgb, gt, best_mask, best_iou, bbox) in zip(axes, comparison_data):
        x, y, bw, bh = bbox
        ax_row[0].imshow(rgb)
        ax_row[0].add_patch(patches.Rectangle(
            (x, y), bw, bh, linewidth=2, edgecolor="red", facecolor="none"))
        ax_row[0].set_title(f"{name}\noriginal + bbox")
        ax_row[0].axis("off")
        ax_row[1].imshow(gt, cmap="gray")
        ax_row[1].set_title("ground truth")
        ax_row[1].axis("off")
        ax_row[2].imshow(best_mask, cmap="gray")
        ax_row[2].set_title(f"predicted\nIoU={best_iou:.3f}")
        ax_row[2].axis("off")
    fig.tight_layout()
    out = os.path.join(plot_dir, "comparison.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
