#pragma once

#include "core/AlphaExpansion.hpp"

template <typename T>
class SequentialStrategy {
public:
    SequentialStrategy(int max_cycles = 100) : max_cycles_(max_cycles) {
    }

    int execute(AlphaExpansion<T> &optimizer, EnergyModel<T> &model) const {
        int num_labels = model.num_labels();
        int cycle = 0;
        bool converged = false;

        while (!converged && cycle < max_cycles_) {
            bool any_changed = false;
            for (int alpha = 0; alpha < num_labels; ++alpha) {
                if (optimizer.perform_expansion_move(alpha)) {
                    any_changed = true;
                }
            }

            if (!any_changed) {
                converged = true;
            }
            cycle++;
        }
        return cycle;
    }

private:
    int max_cycles_;
};
