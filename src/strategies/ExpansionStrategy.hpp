#pragma once

template <typename T>
class AlphaExpansion;

template <typename T>
class EnergyModel;

template <typename T>
class ExpansionStrategy {
public:
    virtual ~ExpansionStrategy() = default;
    virtual int execute(AlphaExpansion<T> &optimizer, EnergyModel<T> &model) const = 0;
};
