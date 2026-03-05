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

    print("Defining Python energy callbacks...")
    LAMBDA = 20
    MAX_UNARY_COST = 50

    def unary_cost(node, label):
        y = node // width
        x = node % width
        d = label

        if x - d < 0:
            return 1000

        sad = int(np.sum(np.abs(left[y, x] - right[y, x - d])))
        return min(sad, MAX_UNARY_COST)

    def pairwise_cost(n1, n2, l1, l2):
        return 0 if l1 == l2 else LAMBDA

    model.set_unary_cost_fn(unary_cost)
    model.set_pairwise_cost_fn(pairwise_cost)

    print(f"Initial energy: {model.evaluate_total_energy()}")

    print("Running Alpha Expansion (BKSolver, SequentialStrategy)...")
    opt = ae.AlphaExpansion(model, "bk")
    strategy = ae.SequentialStrategy(20)
    strategy.execute(opt, model)

    print(f"Final energy: {model.evaluate_total_energy()}")

    labels = np.array(model.get_labels(), dtype=np.uint8).reshape((height, width))

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "tsukuba", "output.png"
    )
    Image.fromarray(labels * 16).save(out_path)
    print(f"Saved disparity map to {out_path}")


if __name__ == "__main__":
    main()
