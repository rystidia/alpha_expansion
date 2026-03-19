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
