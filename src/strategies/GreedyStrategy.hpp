#pragma once

#include "core/AlphaExpansion.hpp"
#include "strategies/ExpansionStrategy.hpp"
#include <type_traits>

/// @brief Expansion strategy that always picks the label with the greatest energy reduction.
///
/// Each cycle evaluates all labels and applies only the expansion move that yields the
/// largest energy decrease. This is more expensive per cycle than `SequentialStrategy`
/// (O(K) max-flow solves per cycle instead of one) but may converge in fewer cycles on
/// some instances.
///
/// @tparam T Numeric cost type.
template <typename T>
class GreedyStrategy : public ExpansionStrategy<T> {
public:
    /// @brief Constructs the strategy.
    /// @param max_cycles Maximum number of greedy cycles before stopping (default: 100).
    GreedyStrategy(int max_cycles = 100) : max_cycles_(max_cycles) {}

    /// @brief Runs greedy alpha-expansion until convergence or `max_cycles`.
    /// @return Number of full cycles completed.
    int execute(AlphaExpansion<T> &optimizer, EnergyModel<T> &model) const override {
        int num_labels = model.num_labels();
        int cycle = 0;
        bool converged = false;

        while (!converged && cycle < max_cycles_) {
            T best_energy = model.evaluate_total_energy();
            int best_alpha = -1;
            std::vector<int> best_labels = model.get_labels();
            std::vector<int> current_labels = model.get_labels();

            for (int alpha = 0; alpha < num_labels; ++alpha) {
                model.set_labels(current_labels);
                if (optimizer.perform_expansion_move(alpha)) {
                    T new_energy = model.evaluate_total_energy();
                    bool improved = false;
                    if constexpr (std::is_floating_point_v<T>) {
                        improved = (best_energy - new_energy > static_cast<T>(1e-5));
                    } else {
                        improved = (new_energy < best_energy);
                    }
                    if (improved) {
                        best_energy = new_energy;
                        best_alpha = alpha;
                        best_labels = model.get_labels();
                    }
                }
            }

            if (best_alpha != -1) {
                model.set_labels(best_labels);
            } else {
                model.set_labels(current_labels);
                converged = true;
            }
            cycle++;
        }
        return cycle;
    }

private:
    int max_cycles_;
};
