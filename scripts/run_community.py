import sys
import os
import networkx as nx
import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))

import alpha_expansion_py as ae


def run_community_detection(
    G, name, labels_dict, args, lambda_val=10, ground_truth=None, visualize=False
):
    print(f"\n{'=' * 50}")
    print(f"Loading {name} graph")

    num_nodes = G.number_of_nodes()
    num_labels = len(set(labels_dict.values()))

    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    print(f"Graph loaded with {num_nodes} nodes and {G.number_of_edges()} edges")

    model = ae.EnergyModel(num_nodes, num_labels, "int32")

    for u, v in G.edges():
        model.add_neighbor(node_to_idx[u], node_to_idx[v])

    def pairwise_cost(n1, n2, l1, l2):
        return 0 if l1 == l2 else lambda_val

    def unary_cost(idx, label):
        node = idx_to_node[idx]
        if node in labels_dict:
            return 0 if label == labels_dict[node] else 1000
        return 0

    model.set_unary_cost_fn(unary_cost)
    model.set_pairwise_cost_fn(pairwise_cost)

    print(f"\nInitial energy: {model.evaluate_total_energy()}")

    print(f"Running Alpha Expansion ({args.solver} solver, {args.strategy} strategy)")
    opt = ae.AlphaExpansionInt(model, args.solver)

    if args.strategy == "sequential":
        strategy = ae.SequentialStrategyInt(args.max_cycles)
    elif args.strategy == "greedy":
        strategy = ae.GreedyStrategyInt(args.max_cycles)
    elif args.strategy == "randomized":
        strategy = ae.RandomizedStrategyInt(args.max_cycles)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    cycles = strategy.execute(opt, model)

    final_energy = model.evaluate_total_energy()
    print(f"Algorithm converged in {cycles} cycles.")
    print(f"Final Energy: {final_energy}")

    labels = model.get_labels()

    print(f"\nFinal Split for {name}")
    for label_idx in range(num_labels):
        faction_members = [
            idx_to_node[i] for i, lbl in enumerate(labels) if lbl == label_idx
        ]
        print(
            f"Faction {label_idx} [{len(faction_members)} members]: {faction_members[:10]}{'...' if len(faction_members) > 10 else ''}"
        )

    if ground_truth:
        match_count = sum(
            1 for node in G.nodes() if labels[node_to_idx[node]] == ground_truth(node)
        )
        print(
            f"\nGround Truth Label Accuracy: {match_count}/{num_nodes} ({(match_count / num_nodes) * 100:.1f}%)"
        )

    if visualize:
        plt.figure(figsize=(12, 10))
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            pos = graphviz_layout(G, prog="neato")
        except ImportError:
            print("Warning: nx_pydot or pydot not found, falling back to spring_layout")
            pos = nx.spring_layout(G, seed=42)
        except Exception as e:
            print(f"Warning: Graphviz layout failed ({e}), falling back to spring_layout")
            pos = nx.spring_layout(G, seed=42)

        color_palette = list(mcolors.TABLEAU_COLORS.values())
        node_colors = [color_palette[labels[i] % len(color_palette)] for i in range(num_nodes)]
        
        edge_colors = ['#CCCCCC'] * num_nodes
        edge_widths = [1.0] * num_nodes
        if ground_truth:
            for i, node in enumerate(G.nodes()):
                if labels[i] != ground_truth(node):
                    edge_colors[i] = '#FF0000'
                    edge_widths[i] = 3.0
        
        nx.draw_networkx(
            G, 
            pos, 
            node_color=node_colors, 
            with_labels=True, 
            node_size=600, 
            font_size=8,
            edge_color='#CCCCCC',
            edgecolors=edge_colors,
            linewidths=edge_widths,
            alpha=0.9
        )
        
        title = f"{name} Communities\nSolver: {args.solver}, Strategy: {args.strategy}, Lambda: {lambda_val}"
        plt.title(title)
        
        safe_name = name.lower().replace(" ", "_").replace("'", "")
        plot_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "community_plots", f"{safe_name}.png"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {os.path.abspath(plot_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Community Detection using Alpha Expansion"
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["bk", "ortools"],
        default="bk",
        help="MaxFlow solver to use",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["sequential", "greedy", "randomized"],
        default="sequential",
        help="Alpha Expansion strategy",
    )
    parser.add_argument(
        "--max_cycles", type=int, default=100, help="Maximum number of expansion cycles"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate and save community plots"
    )
    args = parser.parse_args()

    G_karate = nx.karate_club_graph()
    karate_seeds = {0: 0, 33: 1}

    def karate_gt(n):
        return 0 if G_karate.nodes[n]["club"] == "Mr. Hi" else 1

    run_community_detection(
        G_karate,
        "Zachary's Karate Club",
        karate_seeds,
        args,
        lambda_val=10,
        ground_truth=karate_gt,
        visualize=args.visualize
    )

    G_lesmis = nx.les_miserables_graph()
    lesmis_seeds = {
        "Valjean": 0,
        "Cosette": 0,
        "Marius": 0,
        "Fauchelevent": 0,
        "Javert": 1,
        "Enjolras": 1,
        "Gavroche": 1,
        "Combeferre": 1,
        "Thenardier": 2,
        "MmeThenardier": 2,
        "Boulatruelle": 2,
    }
    run_community_detection(
        G_lesmis, "Les Misérables", lesmis_seeds, args, lambda_val=8, visualize=args.visualize
    )


if __name__ == "__main__":
    main()
