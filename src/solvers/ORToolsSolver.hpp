#pragma once

#include "solvers/MaxFlowSolver.hpp"
#include "ortools/graph/max_flow.h"
#include <vector>
#include <cassert>

/// @brief `MaxFlowSolver` backed by Google OR-Tools `SimpleMaxFlow`.
///
/// Available only when the library is built with `-DUSE_OR_TOOLS=ON` (the default).
///
/// @tparam T Numeric cost type. Note: OR-Tools uses integer arc capacities internally,
///           so floating-point values get truncated inside the flow network.
template <typename T>
class ORToolsSolver : public MaxFlowSolver<T> {
    operations_research::SimpleMaxFlow max_flow_;
    int num_vars_;
    int source_node_;
    int sink_node_;
    T e_const_;
    std::vector<bool> in_source_cut_;

public:
    ORToolsSolver() : num_vars_(0), e_const_(0) {
        source_node_ = 0;
        sink_node_ = 1;
    }

    ~ORToolsSolver() override = default;

    typename MaxFlowSolver<T>::Var add_variable() override {
        num_vars_++;
        sink_node_ = num_vars_ + 1;
        return num_vars_;
    }

    void add_constant(T E) override {
        e_const_ += E;
    }

    void add_edge(int i, int j, T cap, T rev_cap) {
        if (cap > 0) max_flow_.AddArcWithCapacity(i, j, cap);
        if (rev_cap > 0) max_flow_.AddArcWithCapacity(j, i, rev_cap);
    }

    void add_tweights(int i, T cap_source, T cap_sink) {
        T delta = cap_source < cap_sink ? cap_source : cap_sink;
        if (delta < 0) {
            e_const_ += delta;
            cap_source -= delta;
            cap_sink -= delta;
        }
        if (cap_source > 0) max_flow_.AddArcWithCapacity(source_node_, i, static_cast<operations_research::SimpleMaxFlow::FlowQuantity>(cap_source));
        if (cap_sink > 0) max_flow_.AddArcWithCapacity(i, sink_node_, static_cast<operations_research::SimpleMaxFlow::FlowQuantity>(cap_sink));
    }

    void add_term1(typename MaxFlowSolver<T>::Var x, T E0, T E1) override {
        this->add_tweights(x, E1, E0);
    }

    void add_term2(typename MaxFlowSolver<T>::Var x, typename MaxFlowSolver<T>::Var y, T A, T B, T C, T D) override {
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

    T minimize() override {
        max_flow_.Solve(source_node_, sink_node_);
        std::vector<int> source_cut;
        max_flow_.GetSourceSideMinCut(&source_cut);
        in_source_cut_.assign(num_vars_ + 2, false);
        for (int node: source_cut) {
            in_source_cut_[node] = true;
        }

        return e_const_ + static_cast<T>(max_flow_.OptimalFlow());
    }

    int get_var(typename MaxFlowSolver<T>::Var x) override {
        return in_source_cut_[x] ? 0 : 1;
    }
};
