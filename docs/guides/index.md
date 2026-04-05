# Alpha Expansion Library

A C++ library for the Alpha-Expansion graph-cut algorithm, with Python bindings.
Alpha-expansion is used to minimize discrete Markov Random Field (MRF) energies
and is commonly applied in image segmentation and community detection.

## Features

- Three expansion strategies: **Sequential**, **Greedy** and **Randomized**
- Two max-flow backends: **Boykov-Kolmogorov** (default) and **Google OR-Tools**
- Fully templated on cost type (`int32`, `float`, `double`)
- Python bindings via pybind11
- Extensible: plug in custom solvers and strategies

## Guides

- [Getting Started](getting-started.md): build the library and run your first optimization
- [Architecture](architecture.md): how the four main classes fit together
- [Custom Solver](custom-solver.md): implement a new max-flow backend
- [Custom Strategy](custom-strategy.md): implement a new expansion strategy
- [Python API](python-api.md): using the library from Python
- [Demo App](demo-app.md): interactive visualization of the algorithm

## API Reference

See the class list in the sidebar for the full auto-generated API reference.
