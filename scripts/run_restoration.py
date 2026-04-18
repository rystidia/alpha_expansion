import argparse, csv, os, sys
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import load_restoration_image, add_noise
from experiments import run_one
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
import alpha_expansion_py as ae

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0.0 else 10.0 * np.log10(255.0 ** 2 / mse)


def build_model(noisy: np.ndarray, num_labels: int, lambda_smooth: int):
    h, w = noisy.shape
    levels = np.linspace(0, 255, num_labels).astype(np.float32)
    diff = noisy.astype(np.float32)[..., None] - levels[None, None, :]
    unary = (diff * diff).astype(np.int32)
    model = ae.EnergyModel(h * w, num_labels, "int32")
    model.set_unary_costs(unary.flatten().tolist())
    pairwise = np.full((num_labels, num_labels), lambda_smooth, dtype=np.int32)
    np.fill_diagonal(pairwise, 0)
    model.set_pairwise_costs(pairwise.flatten().tolist())
    model.add_grid_edges(w, h)
    return model, levels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="airplane,house,peppers")
    p.add_argument("--num-labels", type=int, default=256)
    p.add_argument("--sigma", type=float, default=20.0)
    p.add_argument("--lambda-smooth", type=int, default=750)
    p.add_argument("--strategies", default="sequential,greedy,randomized")
    p.add_argument("--solvers", default="bk,ortools")
    p.add_argument("--max-cycles", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(ROOT, "data", "results", "restoration")
    plot_dir = os.path.join(ROOT, "data", "plots", "restoration")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    rows = []
    comparison_data = []
    for name in args.images.split(","):
        clean = load_restoration_image(name)
        noisy = add_noise(clean, args.sigma)
        Image.fromarray(noisy).save(os.path.join(plot_dir, f"{name}_noisy.png"))
        best_restored, best_psnr = None, -1.0
        for strategy in args.strategies.split(","):
            for solver in args.solvers.split(","):
                model, levels = build_model(noisy, args.num_labels, args.lambda_smooth)
                res = run_one(model, strategy, solver, args.max_cycles)
                labels = np.array(model.get_labels()).reshape(noisy.shape)
                restored = levels[labels].astype(np.uint8)
                Image.fromarray(restored).save(
                    os.path.join(plot_dir, f"{name}_{strategy}_{solver}.png"))
                p_noisy = psnr(clean, noisy)
                p_restored = psnr(clean, restored)
                if p_restored > best_psnr:
                    best_psnr, best_restored = p_restored, restored
                res.update(image=name, strategy=strategy, solver=solver,
                           psnr_noisy=p_noisy, psnr_restored=p_restored)
                rows.append(res)
                print(f"{name:10} {strategy:10} {solver:8} cycles={res['cycles']:3} "
                      f"PSNR {p_noisy:.2f}->{p_restored:.2f}")
        comparison_data.append((name, clean, noisy, best_restored, best_psnr))

    csv_path = os.path.join(out_dir, "results.csv")
    cols = ["image", "strategy", "solver", "cycles", "moves_applied",
            "initial_energy", "final_energy", "wall_seconds",
            "psnr_noisy", "psnr_restored"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"wrote {csv_path}")
    _comparison_plot(comparison_data, plot_dir)


def _comparison_plot(comparison_data, plot_dir):
    import matplotlib.pyplot as plt
    n_images = len(comparison_data)
    fig, axes = plt.subplots(n_images, 3, figsize=(10, 3.5 * n_images))
    if n_images == 1:
        axes = [axes]
    col_titles = ["Original", "Noisy", "Restored (best)"]
    for ax_row, (name, clean, noisy, restored, best_psnr) in zip(axes, comparison_data):
        imgs = [clean, noisy, restored]
        psnr_noisy = psnr(clean, noisy)
        subtitles = [
            name,
            f"PSNR {psnr_noisy:.2f} dB",
            f"PSNR {best_psnr:.2f} dB",
        ]
        for ax, img, col_title, subtitle in zip(ax_row, imgs, col_titles, subtitles):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{col_title}\n{subtitle}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    out = os.path.join(plot_dir, "comparison.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
