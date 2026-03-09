import sys
import os
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))

import alpha_expansion_py as ae


def main():
    print("Loading Zachary's Karate Club graph")
    G = nx.karate_club_graph()

    num_nodes = G.number_of_nodes()
    num_labels = 2  # Faction 0 (Mr. Hi) and Faction 1 (Officer)

    print(
        f"Graph loaded with {num_nodes} members and {G.number_of_edges()} friendships"
    )

    model = ae.EnergyModel(num_nodes, num_labels)

    for u, v in G.edges():
        model.add_neighbor(u, v)

    LAMBDA = 10

    def pairwise_cost(n1, n2, l1, l2):
        return 0 if l1 == l2 else LAMBDA

    # Enforce two faction leaders to stay in their factions
    def unary_cost(node, label):
        if node == 0:
            return 0 if label == 0 else 1000
        elif node == 33:
            return 0 if label == 1 else 1000
        else:
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
    mr_hi_faction = [n for n, label in enumerate(labels) if label == 0]
    officer_faction = [n for n, label in enumerate(labels) if label == 1]

    print("\nFinal Split")
    print(
        f"Mr. Hi's Faction (Label 0)   [{len(mr_hi_faction)} members]: {mr_hi_faction}"
    )
    print(
        f"Officer's Faction (Label 1) [{len(officer_faction)} members]: {officer_faction}"
    )

    match_count = sum(
        1
        for n in G.nodes()
        if labels[n] == (0 if G.nodes[n]["club"] == "Mr. Hi" else 1)
    )
    print(
        f"\nGround Truth Label Accuracy: {match_count}/{num_nodes} ({(match_count / num_nodes) * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
