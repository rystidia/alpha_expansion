#include <gtest/gtest.h>
#include "core/EnergyModel.hpp"

#include <vector>
#include <cmath>

TEST(EnergyModelTest, UnaryDefaultsToZero) {
    EnergyModel<int> model(3, 2);
    for (int n = 0; n < 3; ++n)
        for (int l = 0; l < 2; ++l)
            EXPECT_EQ(model.get_unary_cost(n, l), 0);
}

TEST(EnergyModelTest, UnaryCostCallback) {
    EnergyModel<int> model(3, 2);
    model.set_unary_cost_fn([](int node, int label) -> int {
        return node * 10 + label;
    });
    EXPECT_EQ(model.get_unary_cost(0, 0), 0);
    EXPECT_EQ(model.get_unary_cost(0, 1), 1);
    EXPECT_EQ(model.get_unary_cost(2, 1), 21);
}

TEST(EnergyModelTest, UnaryCostDenseArray) {
    EnergyModel<int> model(2, 3);
    model.set_unary_costs({10, 20, 30, 40, 50, 60});
    EXPECT_EQ(model.get_unary_cost(0, 0), 10);
    EXPECT_EQ(model.get_unary_cost(0, 2), 30);
    EXPECT_EQ(model.get_unary_cost(1, 1), 50);
}

TEST(EnergyModelTest, UnaryCostArrayOverridesCallback) {
    EnergyModel<int> model(2, 2);
    model.set_unary_cost_fn([](int, int) -> int { return 999; });
    model.set_unary_costs({1, 2, 3, 4});
    // array must take priority
    EXPECT_EQ(model.get_unary_cost(0, 0), 1);
    EXPECT_EQ(model.get_unary_cost(1, 1), 4);
}

TEST(EnergyModelTest, UnaryCostArrayWrongSizeThrows) {
    EnergyModel<int> model(2, 3);
    EXPECT_THROW(model.set_unary_costs({1, 2, 3}), std::invalid_argument);
}

TEST(EnergyModelTest, PairwiseDefaultsToZero) {
    EnergyModel<int> model(3, 2);
    model.add_neighbor(0, 1);
    EXPECT_EQ(model.get_pairwise_cost(0, 1, 0, 1), 0);
}

TEST(EnergyModelTest, PairwiseCostCallback) {
    EnergyModel<int> model(3, 2);
    model.set_pairwise_cost_fn([](int n1, int n2, int l1, int l2) -> int {
        return l1 == l2 ? 0 : 7;
    });
    EXPECT_EQ(model.get_pairwise_cost(0, 1, 0, 0), 0);
    EXPECT_EQ(model.get_pairwise_cost(0, 1, 0, 1), 7);
}

TEST(EnergyModelTest, PairwiseCostDenseMatrix) {
    EnergyModel<int> model(2, 3);
    model.set_pairwise_costs({0, 5, 5, 5, 0, 5, 5, 5, 0});
    EXPECT_EQ(model.get_pairwise_cost(0, 1, 0, 0), 0);
    EXPECT_EQ(model.get_pairwise_cost(0, 1, 1, 2), 5);
    EXPECT_EQ(model.get_pairwise_cost(0, 1, 2, 2), 0);
}

TEST(EnergyModelTest, PairwiseDenseOverridesCallback) {
    EnergyModel<int> model(2, 2);
    model.set_pairwise_cost_fn([](int, int, int, int) -> int { return 999; });
    model.set_pairwise_costs({0, 10, 10, 0});
    EXPECT_EQ(model.get_pairwise_cost(0, 1, 0, 1), 10);
}

TEST(EnergyModelTest, PairwiseDenseWrongSizeThrows) {
    EnergyModel<int> model(2, 3);
    EXPECT_THROW(model.set_pairwise_costs({0, 1}), std::invalid_argument);
}

TEST(EnergyModelTest, EdgeWeightsOverrideDenseAndCallback) {
    EnergyModel<int> model(3, 2);
    model.set_pairwise_cost_fn([](int, int, int, int) -> int { return 999; });
    model.set_pairwise_costs({0, 50, 50, 0});
    model.add_neighbor(0, 1);
    model.add_neighbor(1, 2);
    model.set_edge_weights({0}, {1}, {77});
    EXPECT_EQ(model.get_pairwise_cost(0, 1, 0, 0), 0);
    EXPECT_EQ(model.get_pairwise_cost(0, 1, 0, 1), 77);
    EXPECT_EQ(model.get_pairwise_cost(1, 2, 0, 1), 50);
}

TEST(EnergyModelTest, EdgeWeightsVectorSizeMismatchThrows) {
    EnergyModel<int> model(3, 2);
    EXPECT_THROW(model.set_edge_weights({0, 1}, {1}, {5, 5}), std::invalid_argument);
}

TEST(EnergyModelTest, AddNeighborBidirectional) {
    EnergyModel<int> model(3, 2);
    model.add_neighbor(0, 2);
    const auto& neighbors0 = model.get_neighbors(0);
    const auto& neighbors2 = model.get_neighbors(2);
    ASSERT_EQ(neighbors0.size(), 1u);
    EXPECT_EQ(neighbors0[0], 2);
    ASSERT_EQ(neighbors2.size(), 1u);
    EXPECT_EQ(neighbors2[0], 0);
    EXPECT_TRUE(model.get_neighbors(1).empty());
}

TEST(EnergyModelTest, AddGridEdges4Connected) {
    EnergyModel<int> model(6, 2);  // 3x2 grid
    model.add_grid_edges(3, 2);
    // Corner (0,0) = node 0: neighbors are 1 (right) and 3 (down)
    auto n0 = model.get_neighbors(0);
    EXPECT_EQ(n0.size(), 2u);
    // Center-ish node 1: neighbors 0 (left), 2 (right), 4 (down)
    auto n1 = model.get_neighbors(1);
    EXPECT_EQ(n1.size(), 3u);
    // Interior node 4: neighbors 3 (left), 5 (right), 1 (up)
    auto n4 = model.get_neighbors(4);
    EXPECT_EQ(n4.size(), 3u);
}

TEST(EnergyModelTest, AddGridEdgesWrongSizeThrows) {
    EnergyModel<int> model(6, 2);
    EXPECT_THROW(model.add_grid_edges(4, 2), std::invalid_argument);
}

TEST(EnergyModelTest, LabelsInitToZero) {
    EnergyModel<int> model(5, 3);
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(model.get_label(i), 0);
}

TEST(EnergyModelTest, SetAndGetLabels) {
    EnergyModel<int> model(3, 4);
    model.set_label(0, 2);
    model.set_label(1, 3);
    model.set_label(2, 0);
    EXPECT_EQ(model.get_label(0), 2);
    EXPECT_EQ(model.get_label(1), 3);
    EXPECT_EQ(model.get_label(2), 0);
}

TEST(EnergyModelTest, SetAndGetLabelsBulk) {
    EnergyModel<int> model(4, 3);
    model.set_labels({1, 2, 0, 1});
    auto labels = model.get_labels();
    EXPECT_EQ(labels, (std::vector<int>{1, 2, 0, 1}));
}

TEST(EnergyModelTest, ActiveNodesExcludesAlphaLabel) {
    EnergyModel<int> model(5, 3);
    model.set_labels({0, 1, 2, 0, 1});
    auto active = model.get_active_nodes(0);
    // Nodes 1, 2, 4 do not have label 0
    EXPECT_EQ(active.size(), 3u);
    EXPECT_EQ(active[0], 1);
    EXPECT_EQ(active[1], 2);
    EXPECT_EQ(active[2], 4);
}

TEST(EnergyModelTest, ActiveNodesAllActiveWhenNoneMatch) {
    EnergyModel<int> model(3, 4);
    // All labels are 0, ask for alpha=3
    auto active = model.get_active_nodes(3);
    EXPECT_EQ(active.size(), 3u);
}

TEST(EnergyModelTest, EvaluateEnergyUnaryOnly) {
    EnergyModel<int> model(3, 2);
    model.set_unary_costs({10, 20, 30, 40, 50, 60});
    model.set_labels({0, 1, 0});
    // E = cost(0,0) + cost(1,1) + cost(2,0) = 10 + 40 + 50
    EXPECT_EQ(model.evaluate_total_energy(), 100);
}

TEST(EnergyModelTest, EvaluateEnergyWithPairwise) {
    EnergyModel<int> model(3, 2);
    model.set_unary_cost_fn([](int, int) -> int { return 0; });
    model.set_pairwise_cost_fn([](int, int, int l1, int l2) -> int {
        return l1 == l2 ? 0 : 5;
    });
    model.add_neighbor(0, 1);
    model.add_neighbor(1, 2);
    model.set_labels({0, 1, 1});
    // Pair (0,1): different labels -> 5
    // Pair (1,2): same labels -> 0
    EXPECT_EQ(model.evaluate_total_energy(), 5);
}

TEST(EnergyModelTest, EvaluateEnergyWithProposedLabels) {
    EnergyModel<int> model(2, 2);
    model.set_unary_costs({0, 10, 10, 0});
    model.add_neighbor(0, 1);
    model.set_pairwise_cost_fn([](int, int, int l1, int l2) -> int {
        return l1 == l2 ? 0 : 3;
    });
    model.set_labels({0, 0});
    EXPECT_EQ(model.evaluate_total_energy(), 10);  // unary: 0+10, pair: 0
    // Evaluate with proposed labels without changing model state
    int proposed_energy = model.evaluate_total_energy({1, 1});
    EXPECT_EQ(proposed_energy, 10);  // unary: 10+0, pair: 0
    // Model labels unchanged
    EXPECT_EQ(model.get_label(0), 0);
}

TEST(EnergyModelTest, FloatCosts) {
    EnergyModel<float> model(2, 2);
    model.set_unary_cost_fn([](int node, int label) -> float {
        return node * 1.5f + label * 0.5f;
    });
    EXPECT_FLOAT_EQ(model.get_unary_cost(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(model.get_unary_cost(1, 1), 2.0f);
}

TEST(EnergyModelTest, DoubleCosts) {
    EnergyModel<double> model(2, 2);
    model.set_unary_cost_fn([](int node, int label) -> double {
        return node * 1.5 + label * 0.5;
    });
    EXPECT_DOUBLE_EQ(model.get_unary_cost(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(model.get_unary_cost(1, 1), 2.0);
}
