import sys
import os
import numpy as np
import argparse
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))

import alpha_expansion_py as ae
from load_dataset import load_tsukuba

def main():
    parser = argparse.ArgumentParser(
        description="Trace Alpha Expansion moves on an image"
    )
    parser.add_argument(
        "--max_cycles", type=int, default=3, help="Maximum number of expansion cycles"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/trace_output", help="Directory to save frames"
    )
    args = parser.parse_args()

    left, right, gt = load_tsukuba()
    height, width, _ = left.shape
    num_nodes = height * width
    num_labels = 16

    os.makedirs(args.output_dir, exist_ok=True)

    model = ae.EnergyModel(num_nodes, num_labels, "int32")

    for y in range(height):
        for x in range(width):
            node = y * width + x
            if x + 1 < width: model.add_neighbor(node, node + 1)
            if y + 1 < height: model.add_neighbor(node, node + width)

    LAMBDA = 20
    MAX_UNARY_COST = 50
    
    pairwise_costs = np.full((num_labels, num_labels), LAMBDA, dtype=np.int32)
    np.fill_diagonal(pairwise_costs, 0)
    model.set_pairwise_costs(pairwise_costs.flatten().tolist())

    left_int = left.astype(np.int32)
    right_int = right.astype(np.int32)
    unary_costs = np.full((height, width, num_labels), 1000, dtype=np.int32)
    for d in range(num_labels):
        if d == 0:
            diff = np.sum(np.abs(left_int - right_int), axis=2)
            unary_costs[:, :, d] = np.minimum(diff, MAX_UNARY_COST)
        else:
            diff = np.sum(np.abs(left_int[:, d:] - right_int[:, :-d]), axis=2)
            unary_costs[:, d:, d] = np.minimum(diff, MAX_UNARY_COST)
    model.set_unary_costs(unary_costs.flatten().tolist())

    model.set_labels([0] * num_nodes)
    optimizer = ae.AlphaExpansionInt(model, "bk")
    
    frame_count = 0
    
    def save_frame(label_vec, title, cycle, alpha):
        nonlocal frame_count
        labels = np.array(label_vec, dtype=np.uint8).reshape((height, width))
        img_data = labels * (255 // (num_labels - 1))
        
        frame_path = os.path.join(args.output_dir, f"frame_{frame_count:04d}.png")
        Image.fromarray(img_data).save(frame_path)
        print(f"Saved {frame_path} (Cycle {cycle}, Alpha {alpha})")
        frame_count += 1

    save_frame(model.get_labels(), "Initial State", 0, -1)

    for cycle in range(1, args.max_cycles + 1):
        changed_in_cycle = False
        for alpha in range(num_labels):
            changed = optimizer.perform_expansion_move(alpha)
            if changed:
                changed_in_cycle = True
                save_frame(model.get_labels(), f"Cycle {cycle}, Alpha {alpha}", cycle, alpha)
        
        if not changed_in_cycle:
            print("Converged.")
            break

    print(f"\nTrace complete. {frame_count} frames saved to {args.output_dir}")

if __name__ == "__main__":
    main()
