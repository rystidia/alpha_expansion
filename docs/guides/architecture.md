# Architecture

The library is built around four classes:

<!-- TODO: add architecture diagram -->

## EnergyModel\<T\>

`EnergyModel` owns the graph structure (neighbor lists) and the energy function
(unary and pairwise costs). It also holds the current label assignment, which
strategies update after each move.

The cost type `T` can be `int32_t`, `float` or `double`. Integer costs are generally
preferred for image segmentation (faster and exact). Floating-point works better when
cost functions involve continuous quantities.

## AlphaExpansion\<T\>

`AlphaExpansion` performs a single expansion move. Given a label `alpha`, it:

1. Finds all nodes whose current label is not `alpha` (the *active* nodes).
2. Builds a binary QPBO subgraph: each active node is a binary variable that decides
   whether to switch to `alpha` or keep its current label.
3. Solves the subgraph via the injected `MaxFlowSolver`.
4. Applies the result only if the total energy decreases.

`AlphaExpansion` holds a **non-owning reference** to its `EnergyModel`. The model
must outlive the optimizer.

A **factory function** (`SolverFactory`) is used instead of a single solver instance,
because a new solver must be created for every expansion move.

## MaxFlowSolver\<T\>

`MaxFlowSolver` is a pure virtual interface. The library provides two implementations:

| Class | Backend | Available |
|-------|---------|-----------|
| `BKSolver<T>` | Boykov-Kolmogorov | always |
| `ORToolsSolver<T>` | Google OR-Tools | `-DUSE_OR_TOOLS=ON` |

To use a different backend, implement the virtual methods and pass a factory lambda
to `AlphaExpansion`. See the [Custom Solver guide](custom-solver.md).

## ExpansionStrategy\<T\>

`ExpansionStrategy` is a pure virtual interface that controls the iteration loop.
It calls `AlphaExpansion::perform_expansion_move()` in whatever order it defines
and returns the number of cycles completed.

| Class | Behavior |
|-------|----------|
| `SequentialStrategy<T>` | Labels 0, 1, ..., K-1 in fixed order each cycle |
| `GreedyStrategy<T>` | Best-energy label first each cycle (O(K) solves/cycle) |
| `RandomizedStrategy<T>` | Random shuffle of labels each cycle |

To implement a custom strategy, see the [Custom Strategy guide](custom-strategy.md).
