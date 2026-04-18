import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
import alpha_expansion_py as ae


def build_chain(n: int):
    model = ae.EnergyModel(n, 3, "int32")
    unary = np.zeros((n, 3), dtype=np.int32)
    unary[:, 0] = 3
    for i in range(n):
        if i % 2 == 0:
            unary[i, 1] = 0
            unary[i, 2] = 8
        else:
            unary[i, 1] = 8
            unary[i, 2] = 0
    model.set_unary_costs(unary.flatten().tolist())
    pairwise = np.full((3, 3), 2, dtype=np.int32)
    np.fill_diagonal(pairwise, 0)
    model.set_pairwise_costs(pairwise.flatten().tolist())
    for i in range(n - 1):
        model.add_neighbor(i, i + 1)
    optimum = [1 if i % 2 == 0 else 2 for i in range(n)]
    return model, optimum


def build_checkerboard(side: int):
    n = side * side
    model = ae.EnergyModel(n, 3, "int32")
    unary = np.zeros((n, 3), dtype=np.int32)
    unary[:, 0] = 15
    for y in range(side):
        for x in range(side):
            idx = y * side + x
            if (x + y) % 2 == 0:
                unary[idx, 1], unary[idx, 2] = 0, 40
            else:
                unary[idx, 1], unary[idx, 2] = 40, 0
    model.set_unary_costs(unary.flatten().tolist())
    pairwise = np.full((3, 3), 6, dtype=np.int32)
    np.fill_diagonal(pairwise, 0)
    model.set_pairwise_costs(pairwise.flatten().tolist())
    model.add_grid_edges(side, side)
    optimum = [1 if (i % side + i // side) % 2 == 0 else 2 for i in range(n)]
    return model, optimum


def build_snake(side: int):
    n = side * side
    model = ae.EnergyModel(n, 3, "int32")

    path = []
    for y in range(side):
        row = list(range(side)) if y % 2 == 0 else list(range(side - 1, -1, -1))
        for x in row:
            path.append(y * side + x)

    unary = np.zeros((n, 3), dtype=np.int32)
    unary[:, 0] = 3
    optimum = [0] * n
    for i, node in enumerate(path):
        if i % 2 == 0:
            unary[node, 1], unary[node, 2] = 0, 8
            optimum[node] = 1
        else:
            unary[node, 1], unary[node, 2] = 8, 0
            optimum[node] = 2
    model.set_unary_costs(unary.flatten().tolist())

    path_edges = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if a > b:
            a, b = b, a
        path_edges.add((a, b))

    n1s, n2s, weights = [], [], []
    for y in range(side):
        for x in range(side):
            idx = y * side + x
            if x + 1 < side:
                a, b = idx, idx + 1
                n1s.append(a); n2s.append(b)
                weights.append(2 if (a, b) in path_edges else 0)
            if y + 1 < side:
                a, b = idx, idx + side
                n1s.append(a); n2s.append(b)
                weights.append(2 if (a, b) in path_edges else 0)
    model.add_grid_edges(side, side)
    model.set_edge_weights(n1s, n2s, weights)
    return model, optimum


def init_zero(model):
    return [0] * model.num_nodes


def init_random(model, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, model.num_labels, size=model.num_nodes).tolist()


def init_partial_optimum(model, optimum, fraction: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = model.num_nodes
    k = int(round(fraction * n))
    chosen = rng.choice(n, size=k, replace=False)
    labels = [0] * n
    for i in chosen:
        labels[i] = optimum[i]
    return labels


def make_strategy(name: str, max_cycles: int = 100):
    if name == "sequential":
        return ae.SequentialStrategyInt(max_cycles)
    if name == "greedy":
        return ae.GreedyStrategyInt(max_cycles)
    if name == "randomized":
        return ae.RandomizedStrategyInt(max_cycles)
    raise ValueError(name)


def run_one(model, strategy_name: str, solver_name: str, max_cycles: int = 100,
            init_labels=None):
    if init_labels is not None:
        model.set_labels(init_labels)
    initial_energy = model.evaluate_total_energy()
    optimizer = ae.AlphaExpansionInt(model, solver_name)
    strategy = make_strategy(strategy_name, max_cycles)
    t0 = time.perf_counter()
    cycles = strategy.execute(optimizer, model)
    elapsed = time.perf_counter() - t0
    return {
        "cycles": cycles,
        "initial_energy": initial_energy,
        "final_energy": model.evaluate_total_energy(),
        "wall_seconds": elapsed,
    }
