#include <gtest/gtest.h>
#include "BKSolver.hpp"
#include "ORToolsSolver.hpp"
#include "EnergyModel.hpp"
#include "AlphaExpansion.hpp"
#include "SequentialStrategy.hpp"
#include "GreedyStrategy.hpp"
#include "RandomizedStrategy.hpp"

#include "graph.cpp"
#include "maxflow.cpp"

template class Graph<int, int, int>;
template class Energy<int, int, int>;

#include <vector>
#include <memory>
#include <tuple>
#include <functional>

using SolverParam = std::function<MaxFlowSolver*(int, int)>;
using StrategyParam = std::function<void(AlphaExpansion &, EnergyModel &)>;
using TestParams = std::tuple<SolverParam, StrategyParam>;

class AlphaExpansionTest : public testing::TestWithParam<TestParams> {
protected:
    static AlphaExpansion::SolverFactory get_factory() {
        return [](const int v, const int e) {
            return std::unique_ptr<MaxFlowSolver>(std::get<0>(GetParam())(v, e));
        };
    }

    static void execute_strategy(AlphaExpansion &optimizer, EnergyModel &model) {
        std::get<1>(GetParam())(optimizer, model);
    }
};

TEST_P(AlphaExpansionTest, Test2DGridDenosingMRF) {
    const int W = 5;
    const int H = 5;
    EnergyModel model(W * H, 3);

    std::vector noisy_image(W * H, 0);
    for (int i = 0; i < W * H; ++i) noisy_image[i] = i % 3;

    for (int i = 0; i < W * H; ++i) model.set_label(i, noisy_image[i]);

    model.set_unary_cost_fn([&](int node, int label) -> EnergyValue {
        int x = node % W;
        int y = node / W;
        int preferred_label = 0;
        if (x >= W / 2) preferred_label = 1;
        if (x == W / 2 && y == H / 2) preferred_label = 2;
        return label == preferred_label ? 0 : 100;
    });

    model.set_pairwise_cost_fn([&](int n1, int n2, int l1, int l2) -> EnergyValue {
        return l1 == l2 ? 0 : 20;
    });

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int i = y * W + x;
            if (x < W - 1) model.add_neighbor(i, i + 1);
            if (y < H - 1) model.add_neighbor(i, i + W);
        }
    }

    AlphaExpansion optimizer(model, get_factory());

    EnergyValue start_energy = model.evaluate_total_energy();
    execute_strategy(optimizer, model);

    EnergyValue final_energy = model.evaluate_total_energy();

    EXPECT_LT(final_energy, start_energy);

    int center_node = (H / 2) * W + (W / 2);
    EXPECT_EQ(model.get_label(center_node), 2);
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    AlphaExpansionTest,
    ::testing::Combine(
        ::testing::Values(
            [](int v, int e) -> MaxFlowSolver* { return new BKSolver(v, e); },
            [](int v, int e) -> MaxFlowSolver* { return new ORToolsSolver(); }
        ),
        ::testing::Values(
            [](AlphaExpansion& opt, EnergyModel& mod) { SequentialStrategy s; s.execute(opt, mod); },
            [](AlphaExpansion& opt, EnergyModel& mod) { GreedyStrategy s; s.execute(opt, mod); },
            [](AlphaExpansion& opt, EnergyModel& mod) { RandomizedStrategy s(100, 42); s.execute(opt, mod); }
        )
    )
);
