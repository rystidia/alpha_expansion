import sys
import os
import csv
import networkx as nx
import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))

import numpy as np
import alpha_expansion_py as ae
from datasets import load_community_graph

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_CONFIG = {
    "karate": {
        "seeds": {0: 0, 33: 1},
        "num_labels": 2,
        "lambda_val": 10,
    },
    "lesmis": {
        "seeds": {
            "Valjean": 0, "Cosette": 0, "Marius": 0, "Fauchelevent": 0,
            "Javert": 1, "Enjolras": 1, "Gavroche": 1, "Combeferre": 1,
            "Thenardier": 2, "MmeThenardier": 2, "Boulatruelle": 2,
        },
        "num_labels": 3,
        "lambda_val": 8,
    },
    "football": {
        "seeds": {1: 0, 19: 1, 2: 2, 3: 3, 44: 4, 36: 5, 12: 6, 0: 7, 7: 8, 17: 9, 11: 10, 28: 11},
        "num_labels": 12,
        "lambda_val": 10,
    },
}


def run_one_dataset(name, G, config, args):
    seeds = config["seeds"]
    num_labels = config["num_labels"]
    lambda_val = config["lambda_val"]

    num_nodes = G.number_of_nodes()
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    print(f"\n{'='*50}\n{name}: {num_nodes} nodes, {G.number_of_edges()} edges")

    model = ae.EnergyModel(num_nodes, num_labels, "int32")
    for u, v in G.edges():
        model.add_neighbor(node_to_idx[u], node_to_idx[v])

    seed_nodes_by_label = {}
    for node, lbl in seeds.items():
        seed_nodes_by_label.setdefault(lbl, []).append(node)

    dist = {}
    for lbl, snodes in seed_nodes_by_label.items():
        lengths = nx.multi_source_dijkstra_path_length(G, snodes)
        for node, d in lengths.items():
            dist[(node, lbl)] = d

    max_dist = max(dist.values()) if dist else 1

    def unary_cost(idx, label):
        node = idx_to_node[idx]
        if node in seeds:
            return 0 if label == seeds[node] else 1000
        d = dist.get((node, label), max_dist)
        return int(d * lambda_val)

    model.set_unary_cost_fn(unary_cost)
    model.set_pairwise_cost_fn(lambda n1, n2, l1, l2: 0 if l1 == l2 else lambda_val)

    if args.init == "random":
        rng = np.random.default_rng(42)
        model.set_labels(rng.integers(0, num_labels, num_nodes).tolist())

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
    print(f"Converged in {cycles} cycles. Final energy: {final_energy}")

    labels = model.get_labels()
    for lbl in range(num_labels):
        members = [idx_to_node[i] for i, l in enumerate(labels) if l == lbl]
        print(f"  Community {lbl} [{len(members)}]: {members[:8]}{'...' if len(members) > 8 else ''}")

    return {"dataset": name, "solver": args.solver, "strategy": args.strategy,
            "nodes": num_nodes, "edges": G.number_of_edges(),
            "cycles": cycles, "final_energy": final_energy,
            "labels": labels, "node_to_idx": node_to_idx}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--solver", choices=["bk", "ortools"], default="bk")
    p.add_argument("--strategy", choices=["sequential", "greedy", "randomized"],
                   default="sequential")
    p.add_argument("--max_cycles", type=int, default=100)
    p.add_argument("--datasets", default="karate,lesmis,football")
    p.add_argument("--init", choices=["zero", "random"], default="random")
    p.add_argument("--output-csv", default=None)
    p.add_argument("--visualize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for name in args.datasets.split(","):
        G = load_community_graph(name)
        config = DATASET_CONFIG[name]
        row = run_one_dataset(name, G, config, args)
        rows.append(row)
        if args.visualize:
            _visualize(name, G, row, config, args)

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        cols = ["dataset", "solver", "strategy", "nodes", "edges", "cycles", "final_energy"]
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in cols})
        print(f"\nwrote {args.output_csv}")


def _visualize(name, G, row, config, args):
    labels = row["labels"]
    node_to_idx = row["node_to_idx"]
    plt.figure(figsize=(12, 10))
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(G, prog="neato")
    except Exception:
        pos = nx.spring_layout(G, seed=42)
    palette = list(mcolors.TABLEAU_COLORS.values())
    colors = [palette[labels[node_to_idx[n]] % len(palette)] for n in G.nodes()]
    nx.draw_networkx(G, pos, node_color=colors, with_labels=True,
                     node_size=400, font_size=7, edge_color="#CCCCCC", alpha=0.9)
    plt.title(f"{name}  solver={args.solver}  strategy={args.strategy}")
    plot_dir = os.path.join(ROOT, "data", "plots", "community")
    os.makedirs(plot_dir, exist_ok=True)
    out = os.path.join(plot_dir, f"{name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
