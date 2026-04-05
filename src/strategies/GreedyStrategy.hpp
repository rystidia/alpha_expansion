#pragma once

#include "core/AlphaExpansion.hpp"
#include "strategies/ExpansionStrategy.hpp"
#include <type_traits>

template <typename T>
class GreedyStrategy : public ExpansionStrategy<T> {
public:
    GreedyStrategy(int max_cycles = 100) : max_cycles_(max_cycles) {
    }

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
