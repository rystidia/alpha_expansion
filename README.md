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
