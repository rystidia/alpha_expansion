import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))

import alpha_expansion_py as ae
from load_dataset import load_tsukuba


def main():
    left, right, gt = load_tsukuba()

    height, width, channels = left.shape
    num_nodes = height * width

    left = left.astype(np.int32)
    right = right.astype(np.int32)

    print(f"Image size: {width}x{height} (nodes: {num_nodes})")

    num_labels = 16
    print(f"Number of labels (disparities): {num_labels}")
    model = ae.EnergyModel(num_nodes, num_labels)

    print("Adding 4-connected grid neighbors...")
    for y in range(height):
        for x in range(width):
            node = y * width + x
            if x + 1 < width:
                model.add_neighbor(node, node + 1)
            if y + 1 < height:
                model.add_neighbor(node, node + width)

    print("Defining and precomputing energy arrays...")
    LAMBDA = 20
    MAX_UNARY_COST = 50

    pairwise_costs = np.full((num_labels, num_labels), LAMBDA, dtype=np.int32)
    np.fill_diagonal(pairwise_costs, 0)
    model.set_pairwise_costs(pairwise_costs.flatten().tolist())

    h, w = height, width
    unary_costs = np.full((h, w, num_labels), 1000, dtype=np.int32)

    for d in range(num_labels):
        if d == 0:
            diff = np.sum(np.abs(left - right), axis=2)
            unary_costs[:, :, d] = np.minimum(diff, MAX_UNARY_COST)
        else:
            diff = np.sum(np.abs(left[:, d:] - right[:, :-d]), axis=2)
            unary_costs[:, d:, d] = np.minimum(diff, MAX_UNARY_COST)

    model.set_unary_costs(unary_costs.flatten().tolist())

    print(f"Initial energy: {model.evaluate_total_energy()}")

    print("Running Alpha Expansion (BKSolver, SequentialStrategy)...")
    opt = ae.AlphaExpansion(model, "bk")
    strategy = ae.SequentialStrategy(20)
    cycles = strategy.execute(opt, model)

    print(f"Algorithm converged in {cycles} cycles.")
    print(f"Final energy: {model.evaluate_total_energy()}")

    max_label = num_labels - 1
    gt_disparities = np.clip(gt / 16, 0, max_label).astype(np.int32)
    gt_labels_list = gt_disparities.flatten().tolist()

    gt_energy = model.evaluate_total_energy_with_labels(gt_labels_list)
    print(f"Ground Truth energy: {gt_energy}")

    labels = np.array(model.get_labels(), dtype=np.uint8).reshape((height, width))

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "tsukuba", "output.png"
    )
    Image.fromarray(labels * 16).save(out_path)
    print(f"Saved disparity map to {out_path}")


if __name__ == "__main__":
    main()
