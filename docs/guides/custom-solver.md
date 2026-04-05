# Implementing a Custom Max-Flow Solver

`MaxFlowSolver<T>` is the interface between `AlphaExpansion` and the underlying
graph-cut backend. Subclass it to plug in any max-flow library.

## Interface

```cpp
// src/solvers/MaxFlowSolver.hpp
template <typename T>
class MaxFlowSolver {
public:
    typedef int Var;
    virtual Var  add_variable()                                       = 0;
    virtual void add_constant(T E)                                    = 0;
    virtual void add_term1(Var x, T E0, T E1)                        = 0;
    virtual void add_term2(Var x, Var y, T E00, T E01, T E10, T E11) = 0;
    virtual T    minimize()                                           = 0;
    virtual int  get_var(Var x)                                       = 0;
    virtual ~MaxFlowSolver() = default;
};
```

## Step-by-Step Example

The following example wraps a hypothetical `LibFoo` max-flow library.

### 1. Create the header

```cpp
// src/solvers/FooSolver.hpp
#pragma once

#include "solvers/MaxFlowSolver.hpp"
#include <libfoo/maxflow.h>

template <typename T>
class FooSolver : public MaxFlowSolver<T> {
    libfoo::Graph graph_;
    int num_vars_ = 0;

public:
    FooSolver() = default;

    typename MaxFlowSolver<T>::Var add_variable() override {
        graph_.add_node();
        return num_vars_++;
    }

    void add_constant(T E) override {
        // if LibFoo has no constant term, store it and add it in minimize()
    }

    void add_term1(typename MaxFlowSolver<T>::Var x, T E0, T E1) override {
        // E0 = cost when x=0 (alpha label), E1 = cost when x=1 (keep current label)
        // check your library's source/sink convention
        graph_.add_tweights(x, E1, E0);
    }

    void add_term2(typename MaxFlowSolver<T>::Var x, typename MaxFlowSolver<T>::Var y,
                   T E00, T E01, T E10, T E11) override {
        // AlphaExpansion always satisfies E00 + E11 <= E01 + E10 for Potts energies
        graph_.add_edge(x, y, E01 - E00, E10 - E11);
    }

    T minimize() override {
        return static_cast<T>(graph_.solve());
    }

    int get_var(typename MaxFlowSolver<T>::Var x) override {
        // return 0 if the node is on the source side (= assigned alpha label)
        return graph_.what_segment(x);
    }
};
```

### 2. Pass it via a factory lambda

```cpp
#include "core/EnergyModel.hpp"
#include "core/AlphaExpansion.hpp"
#include "solvers/FooSolver.hpp"
#include "strategies/SequentialStrategy.hpp"

EnergyModel<int> model(100, 3);
// ... set up costs and graph ...

auto factory = [](int v, int e) {
    return std::make_unique<FooSolver<int>>();
};

AlphaExpansion<int> optimizer(model, factory);
SequentialStrategy<int> strategy(50);
strategy.execute(optimizer, model);
```

## Conventions

- **Variable 0 = alpha label, variable 1 = keep current label.** `get_var()` must return
  `0` for nodes that should switch to alpha.
- **`add_term2` receives a submodular matrix.** For Potts energies, `E00 + E11 <= E01 + E10`
  always holds, so you do not need to handle the non-submodular case.
- The factory is called once per expansion move, so allocate per-move state in the constructor.
