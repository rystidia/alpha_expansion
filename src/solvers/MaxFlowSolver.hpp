#pragma once

#include <cstdint>

/// @brief Abstract interface for a binary max-flow / QPBO solver.
///
/// Implementations build a binary energy function term by term using `add_term1`
/// and `add_term2`, then minimize it via max-flow.
///
/// Built-in implementations: `BKSolver` (Boykov–Kolmogorov), `ORToolsSolver` (Google OR-Tools).
/// Implement this interface and add a factory lambda to `AlphaExpansion` to add a custom solver.
///
/// @tparam T Numeric type for energy values (`int32_t`, `float`, or `double`).
template <typename T>
class MaxFlowSolver {
public:
    virtual ~MaxFlowSolver() = default;

    /// Integer handle identifying a binary variable.
    typedef int Var;

    /// @brief Introduces a new binary variable and returns its handle.
    virtual Var add_variable() = 0;

    /// @brief Adds a constant term to the energy function.
    /// @param E Constant to add.
    virtual void add_constant(T E) = 0;

    /// @brief Adds a unary term E(x) where x ∈ {0, 1}.
    /// @param x  Variable handle.
    /// @param E0 Energy contribution when x = 0.
    /// @param E1 Energy contribution when x = 1.
    virtual void add_term1(Var x, T E0, T E1) = 0;

    /// @brief Adds a pairwise term E(x, y) where x, y ∈ {0, 1}.
    ///
    /// The values must satisfy E00 + E11 ≤ E01 + E10 (submodularity requirement).
    /// @param x   First variable handle.
    /// @param y   Second variable handle.
    /// @param E00 Energy when x=0, y=0.
    /// @param E01 Energy when x=0, y=1.
    /// @param E10 Energy when x=1, y=0.
    /// @param E11 Energy when x=1, y=1.
    virtual void add_term2(Var x, Var y, T E00, T E01, T E10, T E11) = 0;

    /// @brief Minimizes the energy and returns the minimum value.
    virtual T minimize() = 0;

    /// @brief Returns the optimal value of variable @p x (0 or 1) after `minimize()`.
    virtual int get_var(Var x) = 0;
};
