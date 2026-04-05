# Python API

The Python bindings expose the full C++ API via [pybind11](https://pybind11.readthedocs.io/).
After building, the module is available as `alpha_expansion_py` inside `build/`.

## Setup

```python
import sys
sys.path.append('build')          # or set PYTHONPATH=build before running
import alpha_expansion_py as ae
```

You can also add `build/` to your `PYTHONPATH` environment variable so you do not
need to do this in every script.

## Creating an Energy Model

Use the `EnergyModel` factory. It returns the right typed object based on `dtype`:

```python
model = ae.EnergyModel(num_nodes=100, num_labels=3, dtype='int32')    # int32_t
model = ae.EnergyModel(num_nodes=100, num_labels=3, dtype='float32')  # float
model = ae.EnergyModel(num_nodes=100, num_labels=3, dtype='float64')  # double
```

You can also instantiate typed classes directly (`ae.EnergyModelInt`, `ae.EnergyModelFloat`
or `ae.EnergyModelDouble`), but the factory is simpler.

## Setting Costs

**Via callbacks** (general graphs, community detection):

```python
model.set_unary_cost_fn(lambda node, label: 0 if seeds.get(node) == label else 1000)
model.set_pairwise_cost_fn(lambda n1, n2, l1, l2: 0 if l1 == l2 else lambda_val)
```

**Via dense arrays** (image grids, faster):

```python
import numpy as np
unary = np.zeros((num_nodes, num_labels), dtype=np.int32)
# ... fill in the costs ...
model.set_unary_costs(unary.flatten().tolist())
```

**Via per-edge weights** (Potts model, zero cost for same label and `weight` otherwise):

```python
model.set_edge_weights(n1s, n2s, weights)  # three int lists of equal length
```

Cost priority: per-edge weights > dense arrays > callbacks.

## Graph Structure

```python
model.add_neighbor(0, 1)                       # single edge
model.add_grid_edges(width=320, height=240)    # full 4-connected grid
```

## Running Optimization

```python
# Typed variants: AlphaExpansionInt, AlphaExpansionFloat, AlphaExpansionDouble
optimizer = ae.AlphaExpansionInt(model, solver_type='bk')  # or 'ortools'

strategy = ae.SequentialStrategyInt(max_cycles=100)
cycles = strategy.execute(optimizer, model)

print(model.get_labels())             # list of ints, one per node
print(model.evaluate_total_energy())  # final energy value
```

## Available Classes

| Class | Description |
|-------|-------------|
| `EnergyModel` | Factory function (preferred) |
| `EnergyModelInt` / `Float` / `Double` | Typed energy model classes |
| `AlphaExpansionInt` / `Float` / `Double` | Optimizer, ties model to a solver backend |
| `SequentialStrategyInt` / `Float` / `Double` | Fixed-order strategy |
| `GreedyStrategyInt` / `Float` / `Double` | Best-first strategy |
| `RandomizedStrategyInt` / `Float` / `Double` | Random-order strategy |

## Step-by-Step Control

You can drive the algorithm one move at a time:

```python
optimizer = ae.AlphaExpansionInt(model, 'bk')
for cycle in range(10):
    changed = False
    for alpha in range(model.num_labels):
        if optimizer.perform_expansion_move(alpha):
            changed = True
    if not changed:
        break   # converged
```
