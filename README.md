# Alpha Expansion Library

This is a C++ library for the Alpha-Expansion algorithm. It is usually used in computer vision for things like image segmentation or stereo matching.

## Features
- Three different expansion strategies: Sequential, Greedy and Randomized.
- Supports different Max-Flow solvers. Right now we have the Boykov-Kolmogorov (BK) solver and Google OR-Tools.
- Python bindings via pybind11.

## Documentation

The full API reference and guides are published at **https://rystidia.github.io/alpha_expansion/**.

Guides available in `docs/guides/`:
- [Getting Started](docs/guides/getting-started.md): build instructions and first examples in C++ and Python
- [Architecture](docs/guides/architecture.md): overview of the four main classes
- [Python API](docs/guides/python-api.md): how to use the Python bindings
- [Demo App](docs/guides/demo-app.md): interactive PyQt6 demo for image segmentation and community detection
- [Custom Solver](docs/guides/custom-solver.md): how to plug in your own max-flow backend
- [Custom Strategy](docs/guides/custom-strategy.md): how to implement a new expansion strategy

## Folders
- `src/`: The main C++ code.
- `tests/`: C++ unit tests (we use GoogleTest).
- `scripts/`: Python scripts for running experiments and producing plots.
- `demo/`: Interactive PyQt6 demo for image segmentation and community detection.

## How to Build

### System Dependencies
You need the Python development headers for the Python bindings. If you plan to build with OR-Tools enabled, you also need bzip2, OpenSSL and zlib development headers, since the OR-Tools binary distribution links against them:
```bash
sudo apt-get install -y python3-dev libbz2-dev libssl-dev zlib1g-dev
```

### Installing OR-Tools
OR-Tools is enabled by default. Download the prebuilt C++ binary for your platform from the [OR-Tools releases page](https://github.com/google/or-tools/releases), then install it to `/opt/ortools`:
```bash
sudo mkdir -p /opt/ortools
sudo tar -xzf /path/to/your/ortools.tar.gz -C /opt/ortools --strip-components=1
```

### Building
```bash
cmake -B build -S . -DCMAKE_PREFIX_PATH=/opt/ortools
cmake --build build -j$(nproc)
```

The `CMAKE_PREFIX_PATH` flag is needed so that OR-Tools' bundled CMake configs for its own dependencies (Abseil, Protobuf, SCIP and others, all shipped under `/opt/ortools/lib/cmake/`) can be found. Without it the configure step fails with "Could not find a package configuration file provided by `absl`" or similar.

If you don't want to install OR-Tools, you can disable it (no extra system deps or prefix path needed):
```bash
cmake -B build -S . -DUSE_OR_TOOLS=OFF
cmake --build build -j$(nproc)
```

## Installing from a Release

Prebuilt artifacts for each release are on the [Releases page](https://github.com/rystidia/alpha_expansion/releases).

### C++ (prebuilt static library)

Download `alpha_expansion-vX.Y.Z-linux.zip` or `-windows.zip`. Each contains the compiled bk solver, headers, and CMake config files. Extract it and point your project at it via `CMAKE_PREFIX_PATH`:

```cmake
find_package(alpha_expansion REQUIRED)
target_link_libraries(my_app PRIVATE alpha_expansion::graph_cuts)
```

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/alpha_expansion-vX.Y.Z-linux
```

### Python (source distribution)

Download `alpha_expansion-vX.Y.Z-sdist.tar.gz` and install it with pip — pip will compile the C++ extension on your machine for whichever Python version you have, so you need a C++ compiler and CMake installed:

```bash
pip install alpha_expansion-0.1.0.tar.gz
```

By default this builds without OR-Tools. To build the OR-Tools solver in (you need OR-Tools installed and findable by CMake first):

```bash
pip install alpha_expansion-0.1.0.tar.gz --config-settings=cmake.define.USE_OR_TOOLS=ON
```

## How to use in Python
First, create a virtual environment and install the needed packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

After you build the project, you can run the example scripts:
```bash
# Stereo, restoration, segmentation, community-detection experiments
python scripts/run_middlebury.py
python scripts/run_restoration.py
python scripts/run_segmentation.py
python scripts/run_community.py

# Interactive demo (image segmentation + community detection)
python -m demo.interactive_segmentation
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

`strategy.execute(...)` is interruptible with Ctrl+C. The signal is checked between expansion moves, so the run aborts after the current move finishes (a single max-flow solve) and `KeyboardInterrupt` is raised in Python.

## How to use in C++
The library is written in C++, so you can include it in your own code. Because the solver and strategy are decoupled, you can also write your own custom Max-Flow solvers or new expansion strategies by implementing the `MaxFlowSolver` and `ExpansionStrategy` interfaces!

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
This library is based on academic research on graph cuts. The alpha-expansion algorithm was introduced by Boykov, Veksler, and Zabih, and the underlying max-flow solver implementation is from Yuri Boykov and Vladimir Kolmogorov. If you use this software for research, please cite:
- *Efficient Approximate Energy Minimization via Graph Cuts.* Y. Boykov, O. Veksler, R.Zabih. IEEE TPAMI 2001.
- *What Energy Functions can be Minimized via Graph Cuts?* V. Kolmogorov, R.Zabih. IEEE TPAMI 2004.
- *An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision.* Yuri Boykov and Vladimir Kolmogorov. IEEE PAMI 2004.
- *Efficiently Solving Dynamic Markov Random Fields Using Graph Cuts.* Pushmeet Kohli and Philip H.S. Torr. ICCV 2005.

## License and Copyright
This project implements the alpha-expansion algorithm and includes the Boykov-Kolmogorov max-flow solver (`src/bk_maxflow_impl`).

The included max-flow solver was developed by Yuri Boykov and Vladimir Kolmogorov. Its license strictly restricts its use to **research purposes only**. Commercial use is not permitted. Therefore, this entire library—including its modifications, wrappers, and Python bindings—can be used and distributed for academic and non-commercial projects only.

If you use this library in your research, you **must** cite the four publications listed in the References section above, as per the original authors' licensing requirements.
