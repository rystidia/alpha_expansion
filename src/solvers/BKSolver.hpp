#pragma once

#include "solvers/MaxFlowSolver.hpp"
#include "bk_maxflow_impl/energy.h"

/// @brief `MaxFlowSolver` backed by the Boykov–Kolmogorov max-flow algorithm.
///
/// This solver wraps the original BK implementation by Yuri Boykov and Vladimir Kolmogorov.
/// It is the default solver and is available in all builds.
///
/// **License note:** The underlying BK implementation is restricted to non-commercial,
/// research use only. See the project README for details.
///
/// @tparam T Numeric cost type (`int32_t`, `float`, or `double`).
template <typename T>
class BKSolver : public MaxFlowSolver<T> {
    typedef Energy<T, T, T> EnergyT;
    EnergyT* e;

public:
    /// @brief Constructs the solver with pre-allocated capacity.
    /// @param var_num_max   Maximum number of binary variables expected.
    /// @param edge_num_max  Maximum number of edges expected (used for memory pre-allocation).
    BKSolver(int var_num_max, int edge_num_max) {
        e = new EnergyT(var_num_max, edge_num_max);
    }

    ~BKSolver() override {
        delete e;
    }

    typename MaxFlowSolver<T>::Var add_variable() override { return e->add_variable(); }
    void add_constant(T E) override { e->add_constant(E); }
    void add_term1(typename MaxFlowSolver<T>::Var x, T E0, T E1) override { e->add_term1(x, E0, E1); }
    void add_term2(typename MaxFlowSolver<T>::Var x, typename MaxFlowSolver<T>::Var y,
                   T E00, T E01, T E10, T E11) override { e->add_term2(x, y, E00, E01, E10, E11); }
    T minimize() override { return e->minimize(); }
    int get_var(typename MaxFlowSolver<T>::Var x) override { return e->get_var(x); }
};
