#pragma once

#include "core/AlphaExpansion.hpp"
#include "strategies/ExpansionStrategy.hpp"
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

/// @brief Expansion strategy that shuffles the label order randomly each cycle.
///
/// Works the same as `SequentialStrategy`, but the label order is randomly shuffled at
/// the start of each cycle using a Mersenne Twister RNG. Good for studying how the label
/// order affects convergence speed.
///
/// @tparam T Numeric cost type.
template <typename T>
class RandomizedStrategy : public ExpansionStrategy<T> {
public:
    /// @brief Constructs the strategy.
    /// @param max_cycles Maximum number of cycles before stopping (default: 100).
    /// @param seed       Seed for the Mersenne Twister RNG (default: 42).
    RandomizedStrategy(int max_cycles = 100, unsigned int seed = 42)
        : max_cycles_(max_cycles), seed_(seed) {}

    /// @brief Runs randomized alpha-expansion until convergence or `max_cycles`.
    /// @return Number of full cycles completed.
    int execute(AlphaExpansion<T> &optimizer, EnergyModel<T> &model) const override {
        int num_labels = model.num_labels();
        int cycle = 0;
        bool converged = false;

        std::vector<int> label_order(num_labels);
        std::iota(label_order.begin(), label_order.end(), 0);
        std::mt19937 g(seed_);

        while (!converged && cycle < max_cycles_) {
            bool any_changed = false;
            std::shuffle(label_order.begin(), label_order.end(), g);
            for (int alpha: label_order) {
                if (optimizer.perform_expansion_move(alpha)) any_changed = true;
            }
            if (!any_changed) converged = true;
            cycle++;
        }
        return cycle;
    }

private:
    int max_cycles_;
    unsigned int seed_;
};
