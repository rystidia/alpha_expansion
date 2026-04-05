#pragma once

#include "core/AlphaExpansion.hpp"
#include "strategies/ExpansionStrategy.hpp"

/// @brief Expansion strategy that cycles through labels in fixed order 0, 1, …, K-1.
///
/// Each cycle tries one expansion move per label. The strategy converges when a full
/// cycle produces no change in any label, or when `max_cycles` is reached.
/// This is the standard strategy from the original Boykov–Veksler–Zabih paper.
///
/// @tparam T Numeric cost type.
template <typename T>
class SequentialStrategy : public ExpansionStrategy<T> {
public:
    /// @brief Constructs the strategy.
    /// @param max_cycles Maximum number of full label cycles before stopping (default: 100).
    SequentialStrategy(int max_cycles = 100) : max_cycles_(max_cycles) {}

    /// @brief Runs sequential alpha-expansion until convergence or `max_cycles`.
    /// @return Number of full cycles completed.
    int execute(AlphaExpansion<T> &optimizer, EnergyModel<T> &model) const override {
        int num_labels = model.num_labels();
        int cycle = 0;
        bool converged = false;

        while (!converged && cycle < max_cycles_) {
            bool any_changed = false;
            for (int alpha = 0; alpha < num_labels; ++alpha) {
                if (optimizer.perform_expansion_move(alpha)) any_changed = true;
            }
            if (!any_changed) converged = true;
            cycle++;
        }
        return cycle;
    }

private:
    int max_cycles_;
};
