#include <gtest/gtest.h>
#include "BKSolver.hpp"
#include "ORToolsSolver.hpp"
#include "EnergyModel.hpp"
#include "AlphaExpansion.hpp"
#include "SequentialStrategy.hpp"

#include "graph.cpp"
#include "maxflow.cpp"

template class Graph<int, int, int>;
template class Energy<int, int, int>;

#include <vector>
#include <memory>
#include <tuple>

class AlphaExpansionTest : public ::testing::TestWithParam<std::function<MaxFlowSolver*(int, int)>> {
protected:
    AlphaExpansion::SolverFactory get_factory() {
        return [this](int v, int e) {
            return std::unique_ptr<MaxFlowSolver>(GetParam()(v, e));
        };
    }
};

TEST_P(AlphaExpansionTest, Test2DGridDenosingMRF) {
    const int W = 5;
    const int H = 5;
    EnergyModel model(W * H, 3);
    
    std::vector<int> noisy_image(W * H, 0);
    for (int i = 0; i < W * H; ++i) noisy_image[i] = i % 3;
    
    model.set_unary_cost_fn([&](int node, int label) -> EnergyValue {
        int x = node % W;
        int y = node / W;
        int preferred_label = 0;
        if (x >= W / 2) preferred_label = 1;
        if (x == W / 2 && y == H / 2) preferred_label = 2;
        return (label == preferred_label) ? 0 : 100;
    });

    model.set_pairwise_cost_fn([&](int n1, int n2, int l1, int l2) -> EnergyValue {
        return (l1 == l2) ? 0 : 20;
    });

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int i = y * W + x;
            if (x < W - 1) model.add_neighbor(i, i + 1);
            if (y < H - 1) model.add_neighbor(i, i + W);
        }
    }

    AlphaExpansion optimizer(model, get_factory());
    SequentialStrategy strategy;

    strategy.execute(optimizer, model);
    
    EnergyValue final_energy = model.evaluate_total_energy();
    
    EnergyValue start_energy = 0;
    for (int i = 0; i < W * H; ++i) {
        start_energy += model.get_unary_cost(i, 0);
        for (int neighbor : model.get_neighbors(i)) {
            if (i < neighbor) {
                start_energy += model.get_pairwise_cost(i, neighbor, 0, 0);
            }
        }
    }

    EXPECT_LT(final_energy, start_energy);
    int center_node = (H / 2) * W + (W / 2);
    EXPECT_EQ(model.get_label(center_node), 2);
}

INSTANTIATE_TEST_SUITE_P(
    MaxFlowSolvers,
    AlphaExpansionTest,
    ::testing::Values(
        [](int v, int e) -> MaxFlowSolver* { return new BKSolver(v, e); },
        [](int v, int e) -> MaxFlowSolver* { return new ORToolsSolver(); }
    )
);
