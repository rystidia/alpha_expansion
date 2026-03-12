#pragma once

#include <cstdint>

template <typename T>
class MaxFlowSolver {
public:
    virtual ~MaxFlowSolver() = default;

    typedef int Var;

    // adds a new binary variable, returning its ID
    virtual Var add_variable() = 0;

    // adds a constant to the energy function
    virtual void add_constant(T E) = 0;

    // adds a unary term E(x)
    virtual void add_term1(Var x, T E0, T E1) = 0;

    // adds a pairwise submodular term E(x, y)
    virtual void add_term2(Var x, Var y, T E00, T E01, T E10, T E11) = 0;

    // minimizes the energy and returns the minimum value
    virtual T minimize() = 0;

    // gets the optimal value of variable x (0 or 1)
    virtual int get_var(Var x) = 0;
};
