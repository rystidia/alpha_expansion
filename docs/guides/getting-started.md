# Getting Started

## System Requirements

- CMake >= 3.15
- A C++17 compiler (GCC >= 9, Clang >= 10, MSVC >= 19.20)
- Python development headers (for the Python bindings)

```bash
sudo apt-get install -y python3-dev
```

## Installing OR-Tools (optional)

OR-Tools is enabled by default. Download the prebuilt C++ binary for your platform from
the [OR-Tools releases page](https://github.com/google/or-tools/releases), then install it:

```bash
sudo mkdir -p /opt/ortools
sudo tar -xzf /path/to/ortools.tar.gz -C /opt/ortools --strip-components=1
```

To build without OR-Tools, pass `-DUSE_OR_TOOLS=OFF` to CMake (see below).

## Building

```bash
cmake -B build -S .
cmake --build build -j$(nproc)
```

CMake finds OR-Tools automatically if it is installed to `/opt/ortools`. To disable it:

```bash
cmake -B build -S . -DUSE_OR_TOOLS=OFF
cmake --build build -j$(nproc)
```

## Running the Tests

```bash
./build/alpha_expansion_tests
```

## First Example: C++

```cpp
#include "core/EnergyModel.hpp"
#include "core/AlphaExpansion.hpp"
#include "solvers/BKSolver.hpp"
#include "strategies/SequentialStrategy.hpp"

int main() {
    // 4 nodes, 2 labels
    EnergyModel<int> model(4, 2);

    // Node 0 prefers label 0, node 3 prefers label 1
    model.set_unary_cost_fn([](int node, int label) -> int {
        if (node == 0) return label == 0 ? 0 : 10;
        if (node == 3) return label == 1 ? 0 : 10;
        return 0;
    });

    // Connect nodes in a chain: 0-1-2-3
    model.add_neighbor(0, 1);
    model.add_neighbor(1, 2);
    model.add_neighbor(2, 3);

    // Penalize adjacent nodes with different labels
    model.set_pairwise_cost_fn([](int, int, int l1, int l2) -> int {
        return l1 == l2 ? 0 : 5;
    });

    // Create optimizer with BK solver
    auto factory = [](int v, int e) {
        return std::make_unique<BKSolver<int>>(v, e);
    };
    AlphaExpansion<int> optimizer(model, factory);

    // Run sequential strategy
    SequentialStrategy<int> strategy(100);
    int cycles = strategy.execute(optimizer, model);

    for (int i = 0; i < 4; ++i)
        printf("node %d -> label %d\n", i, model.get_label(i));
    printf("converged in %d cycles, energy = %d\n", cycles, model.evaluate_total_energy());
    return 0;
}
```

## First Example: Python

```python
import sys
sys.path.append('build')          # make alpha_expansion_py importable
import alpha_expansion_py as ae

# 4 nodes, 2 labels
model = ae.EnergyModel(4, 2, dtype='int32')

# Node 0 prefers label 0, node 3 prefers label 1
model.set_unary_cost_fn(lambda node, label:
    (0 if label == 0 else 10) if node == 0 else
    (0 if label == 1 else 10) if node == 3 else 0
)

# Chain: 0-1-2-3
model.add_neighbor(0, 1)
model.add_neighbor(1, 2)
model.add_neighbor(2, 3)

# Penalize adjacent nodes with different labels
model.set_pairwise_cost_fn(lambda n1, n2, l1, l2: 0 if l1 == l2 else 5)

# Run optimization
optimizer = ae.AlphaExpansionInt(model, 'bk')
strategy  = ae.SequentialStrategyInt(max_cycles=100)
cycles    = strategy.execute(optimizer, model)

print(model.get_labels())           # e.g. [0, 0, 1, 1]
print(f"energy: {model.evaluate_total_energy()}, cycles: {cycles}")
```
