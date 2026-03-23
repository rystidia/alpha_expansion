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
