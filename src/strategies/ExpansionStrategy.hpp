#pragma once

template <typename T>
class AlphaExpansion;

template <typename T>
class EnergyModel;

/// @brief Abstract base class for alpha-expansion iteration strategies.
///
/// A strategy controls the order in which labels are expanded and the stopping condition.
/// The library ships three concrete strategies: `SequentialStrategy`, `GreedyStrategy`
/// and `RandomizedStrategy`. To add a custom strategy, implement this interface.
///
/// @tparam T Numeric cost type. Must match the type used in the associated `EnergyModel<T>`.
///
/// @par Example: custom strategy
/// @code{.cpp}
/// #include "strategies/ExpansionStrategy.hpp"
///
/// template <typename T>
/// class SinglePassStrategy : public ExpansionStrategy<T> {
/// public:
///     int execute(AlphaExpansion<T>& optimizer, EnergyModel<T>& model) const override {
///         for (int alpha = 0; alpha < model.num_labels(); ++alpha)
///             optimizer.perform_expansion_move(alpha);
///         return 1;
///     }
/// };
/// @endcode
template <typename T>
class ExpansionStrategy {
public:
    virtual ~ExpansionStrategy() = default;

    /// @brief Runs the expansion strategy to convergence (or up to a cycle limit).
    /// @param optimizer The `AlphaExpansion` object that executes individual moves.
    /// @param model     The energy model being optimized (labels are updated in-place).
    /// @return Number of full cycles completed.
    virtual int execute(AlphaExpansion<T> &optimizer, EnergyModel<T> &model) const = 0;
};
