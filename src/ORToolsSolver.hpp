#pragma once

#include "MaxFlowSolver.hpp"
#include "ortools/graph/max_flow.h"
#include <vector>
#include <cassert>

// wrapper for the google or-tools max-flow solver
class ORToolsSolver : public MaxFlowSolver {
    operations_research::SimpleMaxFlow max_flow_;
    int num_vars_;
    int source_node_;
    int sink_node_;
    EnergyValue e_const_;
    std::vector<bool> in_source_cut_;

public:
    ORToolsSolver() : num_vars_(0), e_const_(0) {
        source_node_ = 0;
        sink_node_ = 1;
    }

    ~ORToolsSolver() override = default;

    Var add_variable() override {
        num_vars_++;
        sink_node_ = num_vars_ + 1;
        return num_vars_;
    }

    void add_constant(EnergyValue E) override {
        e_const_ += E;
    }

    void add_edge(int i, int j, EnergyValue cap, EnergyValue rev_cap) {
        if (cap > 0) max_flow_.AddArcWithCapacity(i, j, cap);
        if (rev_cap > 0) max_flow_.AddArcWithCapacity(j, i, rev_cap);
    }

    void add_tweights(int i, EnergyValue cap_source, EnergyValue cap_sink) {
        EnergyValue delta = cap_source < cap_sink ? cap_source : cap_sink;
        if (delta < 0) {
            e_const_ += delta;
            cap_source -= delta;
            cap_sink -= delta;
        }
        if (cap_source > 0) max_flow_.AddArcWithCapacity(source_node_, i, cap_source);
        if (cap_sink > 0) max_flow_.AddArcWithCapacity(i, sink_node_, cap_sink);
    }

    void add_term1(Var x, EnergyValue E0, EnergyValue E1) override {
        this->add_tweights(x, E1, E0);
    }

    void add_term2(Var x, Var y, EnergyValue A, EnergyValue B, EnergyValue C, EnergyValue D) override {
        this->add_tweights(x, D, A);
        B -= A;
        C -= D;

        assert(B + C >= 0);
        if (B < 0) {
            this->add_tweights(x, 0, B);
            this->add_tweights(y, 0, -B);
            this->add_edge(x, y, 0, B + C);
        } else if (C < 0) {
            this->add_tweights(x, 0, -C);
            this->add_tweights(y, 0, C);
            this->add_edge(x, y, B + C, 0);
        } else {
            this->add_edge(x, y, B, C);
        }
    }

    EnergyValue minimize() override {
        max_flow_.Solve(source_node_, sink_node_);
        std::vector<int> source_cut;
        max_flow_.GetSourceSideMinCut(&source_cut);
        in_source_cut_.assign(num_vars_ + 2, false);
        for (int node: source_cut) {
            in_source_cut_[node] = true;
        }

        return e_const_ + max_flow_.OptimalFlow();
    }

    int get_var(Var x) override {
        return in_source_cut_[x] ? 0 : 1;
    }
};
