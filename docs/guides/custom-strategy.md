# Implementing a Custom Expansion Strategy

`ExpansionStrategy<T>` controls the iteration loop: which labels to expand, in what
order and when to stop. Subclass it to implement a custom policy.

## Interface

```cpp
// src/strategies/ExpansionStrategy.hpp
template <typename T>
class ExpansionStrategy {
public:
    virtual int execute(AlphaExpansion<T>& optimizer, EnergyModel<T>& model) const = 0;
    virtual ~ExpansionStrategy() = default;
};
```

`execute()` calls `optimizer.perform_expansion_move(alpha)` for whatever labels and
in whatever order the strategy decides. It returns the number of full cycles completed.

## Example: Single-Pass Strategy

Tries each label exactly once and stops:

```cpp
// src/strategies/SinglePassStrategy.hpp
#pragma once

#include "strategies/ExpansionStrategy.hpp"
#include "core/AlphaExpansion.hpp"
#include "core/EnergyModel.hpp"

template <typename T>
class SinglePassStrategy : public ExpansionStrategy<T> {
public:
    int execute(AlphaExpansion<T>& optimizer, EnergyModel<T>& model) const override {
        for (int alpha = 0; alpha < model.num_labels(); ++alpha)
            optimizer.perform_expansion_move(alpha);
        return 1;
    }
};
```

## Using the Custom Strategy

```cpp
#include "core/EnergyModel.hpp"
#include "core/AlphaExpansion.hpp"
#include "solvers/BKSolver.hpp"
#include "strategies/SinglePassStrategy.hpp"

EnergyModel<int> model(100, 3);
// ... set up costs and graph ...

auto factory = [](int v, int e) {
    return std::make_unique<BKSolver<int>>(v, e);
};
AlphaExpansion<int> optimizer(model, factory);

SinglePassStrategy<int> strategy;
int cycles = strategy.execute(optimizer, model);
```

## Tips

- Call `model.evaluate_total_energy()` to check progress between moves.
- Call `model.get_labels()` and `model.set_labels()` to save and restore state.
  `GreedyStrategy` uses this to roll back non-improving moves.
- Return the number of full cycles from `execute()`. The demo app shows this value
  in the convergence dialog.
