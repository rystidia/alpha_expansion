#include <gtest/gtest.h>
#include "BKSolver.hpp"
#include "ORToolsSolver.hpp"
#include "EnergyModel.hpp"
#include "AlphaExpansion.hpp"
#include "SequentialStrategy.hpp"
#include "GreedyStrategy.hpp"
#include "RandomizedStrategy.hpp"

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

TEST_P(AlphaExpansionTest, TwoPixelTrap_StartsAt_0_0) {
    const int n = 2;
    const int k = 3;
    const int m = 1;
    EnergyModel model(n, k);

    // Starting with (0,0) configuration (local minimum trap)
    model.set_label(0, 0);
    model.set_label(1, 0);

    model.set_unary_cost_fn([&](int node, int label) -> EnergyValue {
        if (node == 0) return (label == 0) ? 1 : (label == 1) ? 3 : 0;
        if (node == 1) return (label == 0) ? 1 : (label == 1) ? 0 : 3;
        return 0;
    });

    model.set_pairwise_cost_fn([&](int n1, int n2, int l1, int l2) -> EnergyValue {
        return (l1 == l2) ? 0 : m;
    });

    model.add_neighbor(0, 1);

    auto factory = get_factory();
    AlphaExpansion optimizer(model, factory);
    execute_strategy(optimizer, model);

    EXPECT_EQ(model.evaluate_total_energy(), 2);
}

TEST_P(AlphaExpansionTest, TwoPixelTrap_StartsAt_0_1) {
    const int n = 2;
    const int k = 3;
    const int m = 1;
    EnergyModel model(n, k);

    // Starting with (0,1) configuration (must reach optimal E=1)
    model.set_label(0, 0);
    model.set_label(1, 1);

    model.set_unary_cost_fn([&](int node, int label) -> EnergyValue {
        if (node == 0) return (label == 0) ? 1 : (label == 1) ? 3 : 0;
        if (node == 1) return (label == 0) ? 1 : (label == 1) ? 0 : 3;
        return 0;
    });

    model.set_pairwise_cost_fn([&](int n1, int n2, int l1, int l2) -> EnergyValue {
        return (l1 == l2) ? 0 : m;
    });

    model.add_neighbor(0, 1);

    auto factory = get_factory();
    AlphaExpansion optimizer(model, factory);
    execute_strategy(optimizer, model);

    EXPECT_EQ(model.evaluate_total_energy(), 1);
}

TEST_P(AlphaExpansionTest, TestManyCycles2) {
    const int n = 40;
    const int a = 3;
    const int b = 7;
    const int m = 2;
    const int k = 3;
    EnergyModel model(n, k);

    for (int i = 0; i < n; ++i) model.set_label(i, 0);

    model.set_unary_cost_fn([&](int node, int label) -> EnergyValue {
        if (node % 2 == 0) {
            return (label == 0) ? a : (label == 1) ? 0 : b;
        }
        return (label == 0) ? a : (label == 1) ? b : 0;
    });

    model.set_pairwise_cost_fn([&](int n1, int n2, int l1, int l2) -> EnergyValue {
        return (l1 == l2) ? 0 : m;
    });

    for (int i = 0; i < n - 1; ++i) {
        model.add_neighbor(i, i + 1);
    }

    AlphaExpansion optimizer(model, get_factory());
    execute_strategy(optimizer, model);

    EnergyValue final_energy = model.evaluate_total_energy();
    // optimal is alternating 1 and 2 with unary = 0.
    // smooth = (n-1) * m
    EXPECT_EQ(final_energy, (n - 1) * m);
}

TEST_P(AlphaExpansionTest, TestSnakeMRF) {
    const int w = 16;
    const int h = 16;
    const int num_labels = 3;
    const int num_pixels = w * h;
    EnergyModel model(num_pixels, num_labels);

    for (int i = 0; i < num_pixels; ++i) model.set_label(i, 0);

    auto get_snake_coords = [](int idx, int w, int h, int &x, int &y) {
        int row = idx / w;
        int col = idx % w;
        y = row;
        x = row % 2 == 0 ? col : w - 1 - col;
    };

    std::vector unary_1(num_pixels, 0);
    std::vector unary_2(num_pixels, 0);

    for (int i = 0; i < num_pixels; ++i) {
        int x, y;
        get_snake_coords(i, w, h, x, y);
        int p = y * w + x;
        if (i % 2 == 0) {
            unary_1[p] = 0;
            unary_2[p] = 8;
        } else {
            unary_1[p] = 8;
            unary_2[p] = 0;
        }
    }

    model.set_unary_cost_fn([&](int node, int label) -> EnergyValue {
        if (label == 0) return 3;
        if (label == 1) return unary_1[node];
        return unary_2[node];
    });

    const EnergyValue h_path = 2;

    std::vector hWeights(num_pixels, 0);
    std::vector vWeights(num_pixels, 0);

    for (int i = 0; i < num_pixels - 1; ++i) {
        int x1, y1, x2, y2;
        get_snake_coords(i, w, h, x1, y1);
        get_snake_coords(i + 1, w, h, x2, y2);

        int p1 = y1 * w + x1;
        if (y1 == y2) {
            if (x2 == x1 + 1) hWeights[p1] = h_path;
            else if (x2 == x1 - 1) hWeights[y2 * w + x2] = h_path;
        } else if (x1 == x2) {
            if (y2 == y1 + 1) vWeights[p1] = h_path;
        }
    }

    model.set_pairwise_cost_fn([&](int n1, int n2, int l1, int l2) -> EnergyValue {
        if (l1 == l2) return 0;
        int x1 = n1 % w;
        int y1 = n1 / w;
        int x2 = n2 % w;
        int y2 = n2 / w;

        EnergyValue weight = 0;
        if (y1 == y2) {
            int left = std::min(x1, x2);
            weight = hWeights[y1 * w + left];
        } else if (x1 == x2) {
            int top = std::min(y1, y2);
            weight = vWeights[top * w + x1];
        }
        return weight;
    });

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int i = y * w + x;
            if (x < w - 1) model.add_neighbor(i, i + 1);
            if (y < h - 1) model.add_neighbor(i, i + w);
        }
    }

    AlphaExpansion optimizer(model, get_factory());
    execute_strategy(optimizer, model);

    EnergyValue final_energy = model.evaluate_total_energy();
    EXPECT_EQ(final_energy, 510);
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
            [](AlphaExpansion& opt, EnergyModel& mod) { GreedyStrategy s(1000); s.execute(opt, mod); },
            [](AlphaExpansion& opt, EnergyModel& mod) { RandomizedStrategy s(100, 42); s.execute(opt, mod); }
        )
    )
);
