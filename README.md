# Alpha Expansion Library

This is a C++ library for the Alpha-Expansion algorithm. It is usually used in computer vision for things like image segmentation or stereo matching.

## Features
- Three different expansion strategies: Sequential, Greedy, and Randomized.
- Supports different Max-Flow solvers. Right now we have the Boykov-Kolmogorov (BK) solver and Google OR-Tools.
- Python bindings. This means you can use the library inside your Python scripts.

## Folders
- `src/`: The main C++ code.
- `tests/`: C++ unit tests (we use GoogleTest).
- `scripts/`: Python scripts for running experiments and visualizing results.

## How to Build
To build the C++ tests and the Python bindings, run these commands:
```bash
cmake -B build -S .
cmake --build build -j4
```

Google OR-Tools is enabled by default. If you don't have OR-Tools installed, you can turn it off:
```bash
cmake -B build -S . -DUSE_OR_TOOLS=OFF
cmake --build build -j4
```

## How to use in Python
First, create a virtual environment and install the needed packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

After you build the project, you can run our example scripts:
```bash
# See community detection with graphs
python scripts/run_community.py --visualize

# Watch how alpha-expansion works step-by-step
python scripts/trace_alpha_expansion.py
```

If you want to write your own Python code, just add the build folder to your path:
```python
import sys
sys.path.append('build')
import alpha_expansion_py as ae

# Create a graph with 10 nodes and 3 labels
model = ae.EnergyModel(num_nodes=10, num_labels=3, dtype="int32")

# ... add your node and edge costs here ...

# Choose solver ("bk" or "ortools") and run the strategy
optimizer = ae.AlphaExpansionInt(model, "bk")
strategy = ae.SequentialStrategyInt(max_cycles=100)
cycles = strategy.execute(optimizer, model)
```

## How to use in C++
The library is written in C++, so you can easily include it in your own code. Because the solver and strategy are decoupled, you can also write your own custom Max-Flow solvers or new expansion strategies by implementing the `MaxFlowSolver` and `ExpansionStrategy` interfaces!

Here is a simple example using the built-in ones:

```cpp
#include "core/EnergyModel.hpp"
#include "core/AlphaExpansion.hpp"
#include "solvers/BKSolver.hpp"
#include "strategies/SequentialStrategy.hpp"

// Create a graph with 10 nodes and 3 labels
EnergyModel<int> model(10, 3);

// Tell the optimizer how to create the solver (you can plug in your own here!)
auto solver_factory = [](int v, int e) { 
    return std::make_unique<BKSolver<int>>(v, e); 
};

// Create the optimizer and run the strategy
AlphaExpansion<int> optimizer(model, solver_factory);
SequentialStrategy<int> strategy(100); // or your custom strategy
strategy.execute(optimizer, model);
```

## Running Tests
After building, you can run the unit tests to make sure everything works:
```bash
./build/alpha_expansion_tests
```

## References
This library is based on academic research. The max-flow solver implementation is from Yuri Boykov and Vladimir Kolmogorov. If you use this software for research, please cite:
- *An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision.* Yuri Boykov and Vladimir Kolmogorov. IEEE PAMI 2004.
- *Efficiently Solving Dynamic Markov Random Fields Using Graph Cuts.* Pushmeet Kohli and Philip H.S. Torr. ICCV 2005.
