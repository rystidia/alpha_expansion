import sys
import os
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))

import alpha_expansion_py as ae


def run_community_detection(G, name, labels_dict, lambda_val=10, ground_truth=None):
    print(f"\n{'=' * 50}")
    print(f"Loading {name} graph")

    num_nodes = G.number_of_nodes()
    num_labels = len(set(labels_dict.values()))

    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    print(f"Graph loaded with {num_nodes} nodes and {G.number_of_edges()} edges")

    model = ae.EnergyModel(num_nodes, num_labels)

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

    print("Running Alpha Expansion (BKSolver, SequentialStrategy)")
    opt = ae.AlphaExpansion(model, "bk")
    strategy = ae.SequentialStrategy(10)
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


def main():
    G_karate = nx.karate_club_graph()
    karate_seeds = {0: 0, 33: 1}

    def karate_gt(n):
        return 0 if G_karate.nodes[n]["club"] == "Mr. Hi" else 1

    run_community_detection(
        G_karate,
        "Zachary's Karate Club",
        karate_seeds,
        lambda_val=10,
        ground_truth=karate_gt,
    )

    G_lesmis = nx.les_miserables_graph()
    lesmis_seeds = {
        "Valjean": 0,
        "Cosette": 0,
        "Marius": 0,
        "Javert": 1,
        "Gavroche": 1,
        "Thenardier": 2,
        "MmeThenardier": 2,
    }
    run_community_detection(G_lesmis, "Les Misérables", lesmis_seeds, lambda_val=5)


if __name__ == "__main__":
    main()
