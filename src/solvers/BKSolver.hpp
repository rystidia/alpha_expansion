#pragma once

#include "solvers/MaxFlowSolver.hpp"
#include "bk_maxflow_impl/energy.h"

// wrapper for the Boykov-Kolmogorov max-flow solver
class BKSolver : public MaxFlowSolver {
    typedef Energy<int, int, int> EnergyT;
    EnergyT* e;

public:
    BKSolver(int var_num_max, int edge_num_max) {
        e = new EnergyT(var_num_max, edge_num_max);
    }

    ~BKSolver() override {
        delete e;
    }

    Var add_variable() override {
        return e->add_variable();
    }

    void add_constant(EnergyValue E) override {
        e->add_constant(E);
    }

    void add_term1(Var x, EnergyValue E0, EnergyValue E1) override {
        e->add_term1(x, E0, E1);
    }

    void add_term2(Var x, Var y, EnergyValue E00, EnergyValue E01, EnergyValue E10, EnergyValue E11) override {
        e->add_term2(x, y, E00, E01, E10, E11);
    }

    EnergyValue minimize() override {
        return e->minimize();
    }

    int get_var(Var x) override {
        return e->get_var(x);
    }
};
