#pragma once

#include "core/AlphaExpansion.hpp"

class GreedyStrategy {
public:
    GreedyStrategy(int max_cycles = 100) : max_cycles_(max_cycles) {
    }

    int execute(AlphaExpansion &optimizer, EnergyModel &model) const {
        int num_labels = model.num_labels();
        int cycle = 0; // TODO: calculate cycles
        bool converged = false;

        while (!converged && cycle < max_cycles_) {
            EnergyValue best_energy = model.evaluate_total_energy();
            int best_alpha = -1;
            std::vector<int> best_labels = model.get_labels();
            std::vector<int> current_labels = model.get_labels();

            for (int alpha = 0; alpha < num_labels; ++alpha) {
                model.set_labels(current_labels);

                if (optimizer.perform_expansion_move(alpha)) {
                    EnergyValue new_energy = model.evaluate_total_energy();
                    if (new_energy < best_energy) {
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
