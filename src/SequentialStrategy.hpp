#pragma once

#include "AlphaExpansion.hpp"
#include <iostream>

class SequentialStrategy {
public:
    SequentialStrategy(int max_cycles = 100) : max_cycles_(max_cycles) {}

    void execute(AlphaExpansion& optimizer, EnergyModel& model) {
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
    }

private:
    int max_cycles_;
};
