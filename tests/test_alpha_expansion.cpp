#include <gtest/gtest.h>
#include "solvers/BKSolver.hpp"
#ifdef USE_OR_TOOLS
#include "solvers/ORToolsSolver.hpp"
#endif
#include "core/EnergyModel.hpp"
#include "core/AlphaExpansion.hpp"
#include "strategies/SequentialStrategy.hpp"
#include "strategies/GreedyStrategy.hpp"
#include "strategies/RandomizedStrategy.hpp"

#include <vector>
#include <memory>
#include <tuple>
#include <string>

enum class SolverType { BK, ORTools };
enum class StrategyType { Sequential, Greedy, Randomized };

using TestParams = std::tuple<SolverType, StrategyType>;

class AlphaExpansionTest : public testing::TestWithParam<TestParams> {
};

template <typename T>
auto get_factory(SolverType solver_type) {
    return [solver_type](const int v, const int e) -> std::unique_ptr<MaxFlowSolver<T>> {
        if (solver_type == SolverType::BK) return std::make_unique<BKSolver<T>>(v, e);
#ifdef USE_OR_TOOLS
        else return std::make_unique<ORToolsSolver<T>>();
#else
        else throw std::runtime_error("ORToolsSolver disabled");
#endif
    };
}

template <typename T>
std::pair<std::string, int> execute_strategy(StrategyType strategy_type, AlphaExpansion<T> &optimizer, EnergyModel<T> &model) {
    if (strategy_type == StrategyType::Sequential) {
        SequentialStrategy<T> s;
        return {"Sequential", s.execute(optimizer, model)};
    } else if (strategy_type == StrategyType::Greedy) {
        GreedyStrategy<T> s(1000);
        return {"Greedy", s.execute(optimizer, model)};
    } else {
        RandomizedStrategy<T> s(1000, 42); // 1000 max cycles
        return {"Randomized", s.execute(optimizer, model)};
    }
}

template <typename T>
void run_Test2DGridDenosing(SolverType solver_type, StrategyType strategy_type) {
    const int W = 5;
    const int H = 5;
    EnergyModel<T> model(W * H, 3);

    std::vector<int> noisy_image(W * H, 0);
    for (int i = 0; i < W * H; ++i) noisy_image[i] = i % 3;

    for (int i = 0; i < W * H; ++i) model.set_label(i, noisy_image[i]);

    model.set_unary_cost_fn([&](int node, int label) -> T {
        int x = node % W;
        int y = node / W;
        int preferred_label = 0;
        if (x >= W / 2) preferred_label = 1;
        if (x == W / 2 && y == H / 2) preferred_label = 2;
        return label == preferred_label ? 0 : 100;
    });

    model.set_pairwise_cost_fn([&](int n1, int n2, int l1, int l2) -> T {
        return l1 == l2 ? 0 : 20;
    });

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int i = y * W + x;
            if (x < W - 1) model.add_neighbor(i, i + 1);
            if (y < H - 1) model.add_neighbor(i, i + W);
        }
    }

    AlphaExpansion<T> optimizer(model, get_factory<T>(solver_type));

    T start_energy = model.evaluate_total_energy();
    auto [strategy_name, cycles] = execute_strategy<T>(strategy_type, optimizer, model);

    T final_energy = model.evaluate_total_energy();
    EXPECT_LT(final_energy, start_energy);

    int center_node = (H / 2) * W + (W / 2);
    EXPECT_EQ(model.get_label(center_node), 2);
}

TEST_P(AlphaExpansionTest, Test2DGridDenosingMRF) {
    auto params = GetParam();
    run_Test2DGridDenosing<int>(std::get<0>(params), std::get<1>(params));
    run_Test2DGridDenosing<float>(std::get<0>(params), std::get<1>(params));
    run_Test2DGridDenosing<double>(std::get<0>(params), std::get<1>(params));
}

template <typename T>
void run_TwoPixelTrap_StartsAt_0_0(SolverType solver_type, StrategyType strategy_type) {
    const int n = 2;
    const int k = 3;
    const T m = 1;
    EnergyModel<T> model(n, k);

    model.set_label(0, 0);
    model.set_label(1, 0);

    model.set_unary_cost_fn([&](int node, int label) -> T {
        if (node == 0) return (label == 0) ? 1 : (label == 1) ? 3 : 0;
        if (node == 1) return (label == 0) ? 1 : (label == 1) ? 0 : 3;
        return 0;
    });

    model.set_pairwise_cost_fn([&](int n1, int n2, int l1, int l2) -> T {
        return (l1 == l2) ? 0 : m;
    });

    model.add_neighbor(0, 1);

    AlphaExpansion<T> optimizer(model, get_factory<T>(solver_type));
    execute_strategy<T>(strategy_type, optimizer, model);

    EXPECT_FLOAT_EQ(model.evaluate_total_energy(), 2.0);
}

TEST_P(AlphaExpansionTest, TwoPixelTrap_StartsAt_0_0) {
    auto params = GetParam();
    run_TwoPixelTrap_StartsAt_0_0<int>(std::get<0>(params), std::get<1>(params));
    run_TwoPixelTrap_StartsAt_0_0<float>(std::get<0>(params), std::get<1>(params));
    run_TwoPixelTrap_StartsAt_0_0<double>(std::get<0>(params), std::get<1>(params));
}

template <typename T>
void run_TwoPixelTrap_StartsAt_0_1(SolverType solver_type, StrategyType strategy_type) {
    const int n = 2;
    const int k = 3;
    const T m = 1;
    EnergyModel<T> model(n, k);

    model.set_label(0, 0);
    model.set_label(1, 1);

    model.set_unary_cost_fn([&](int node, int label) -> T {
        if (node == 0) return (label == 0) ? 1 : (label == 1) ? 3 : 0;
        if (node == 1) return (label == 0) ? 1 : (label == 1) ? 0 : 3;
        return 0;
    });

    model.set_pairwise_cost_fn([&](int n1, int n2, int l1, int l2) -> T {
        return (l1 == l2) ? 0 : m;
    });

    model.add_neighbor(0, 1);

    AlphaExpansion<T> optimizer(model, get_factory<T>(solver_type));
    execute_strategy<T>(strategy_type, optimizer, model);

    EXPECT_FLOAT_EQ(model.evaluate_total_energy(), 1.0);
}

TEST_P(AlphaExpansionTest, TwoPixelTrap_StartsAt_0_1) {
    auto params = GetParam();
    run_TwoPixelTrap_StartsAt_0_1<int>(std::get<0>(params), std::get<1>(params));
    run_TwoPixelTrap_StartsAt_0_1<float>(std::get<0>(params), std::get<1>(params));
    run_TwoPixelTrap_StartsAt_0_1<double>(std::get<0>(params), std::get<1>(params));
}

template <typename T>
void run_TestManyCycles2(SolverType solver_type, StrategyType strategy_type) {
    const int n = 40;
    const T a = 3;
    const T b = 7;
    const T m = 2;
    const int k = 3;
    EnergyModel<T> model(n, k);

    for (int i = 0; i < n; ++i) model.set_label(i, 0);

    model.set_unary_cost_fn([&](int node, int label) -> T {
        if (node % 2 == 0) {
            return (label == 0) ? a : (label == 1) ? 0 : b;
        }
        return (label == 0) ? a : (label == 1) ? b : 0;
    });

    model.set_pairwise_cost_fn([&](int n1, int n2, int l1, int l2) -> T {
        return (l1 == l2) ? 0 : m;
    });

    for (int i = 0; i < n - 1; ++i) {
        model.add_neighbor(i, i + 1);
    }

    AlphaExpansion<T> optimizer(model, get_factory<T>(solver_type));
    auto [strategy_name, cycles] = execute_strategy<T>(strategy_type, optimizer, model);

    EXPECT_FLOAT_EQ(model.evaluate_total_energy(), (n - 1) * m);
    
    if (strategy_type == StrategyType::Sequential) EXPECT_EQ(cycles, 12);
    else if (strategy_type == StrategyType::Greedy) EXPECT_EQ(cycles, 22);
    else if (strategy_type == StrategyType::Randomized) EXPECT_GE(cycles, 5); // Randomized cycles can vary slightly due to epsilon checking order
}

TEST_P(AlphaExpansionTest, TestManyCycles2) {
    auto params = GetParam();
    run_TestManyCycles2<int>(std::get<0>(params), std::get<1>(params));
    run_TestManyCycles2<float>(std::get<0>(params), std::get<1>(params));
    run_TestManyCycles2<double>(std::get<0>(params), std::get<1>(params));
}

template <typename T>
void run_TestSnakeMRF(SolverType solver_type, StrategyType strategy_type) {
    const int w = 16;
    const int h = 16;
    const int num_labels = 3;
    const int num_pixels = w * h;
    EnergyModel<T> model(num_pixels, num_labels);

    for (int i = 0; i < num_pixels; ++i) model.set_label(i, 0);

    auto get_snake_coords = [](int idx, int w, int h, int &x, int &y) {
        int row = idx / w;
        int col = idx % w;
        y = row;
        x = row % 2 == 0 ? col : w - 1 - col;
    };

    std::vector<T> unary_1(num_pixels, 0);
    std::vector<T> unary_2(num_pixels, 0);

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

    model.set_unary_cost_fn([&](int node, int label) -> T {
        if (label == 0) return 3;
        if (label == 1) return unary_1[node];
        return unary_2[node];
    });

    const T h_path = 2;

    std::vector<T> hWeights(num_pixels, 0);
    std::vector<T> vWeights(num_pixels, 0);

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

    model.set_pairwise_cost_fn([&](int n1, int n2, int l1, int l2) -> T {
        if (l1 == l2) return 0;
        int x1 = n1 % w;
        int y1 = n1 / w;
        int x2 = n2 % w;
        int y2 = n2 / w;

        T weight = 0;
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

    AlphaExpansion<T> optimizer(model, get_factory<T>(solver_type));
    auto [strategy_name, cycles] = execute_strategy<T>(strategy_type, optimizer, model);

    EXPECT_FLOAT_EQ(model.evaluate_total_energy(), 510.0);
    
    if (strategy_type == StrategyType::Sequential) EXPECT_EQ(cycles, 66);
    else if (strategy_type == StrategyType::Greedy) EXPECT_EQ(cycles, 130);
    else if (strategy_type == StrategyType::Randomized) EXPECT_GE(cycles, 50);
}

TEST_P(AlphaExpansionTest, TestSnakeMRF) {
    auto params = GetParam();
    run_TestSnakeMRF<int>(std::get<0>(params), std::get<1>(params));
    run_TestSnakeMRF<float>(std::get<0>(params), std::get<1>(params));
    run_TestSnakeMRF<double>(std::get<0>(params), std::get<1>(params));
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    AlphaExpansionTest,
    ::testing::Combine(
#ifdef USE_OR_TOOLS
        ::testing::Values(SolverType::BK, SolverType::ORTools),
#else
        ::testing::Values(SolverType::BK),
#endif
        ::testing::Values(StrategyType::Sequential, StrategyType::Greedy, StrategyType::Randomized)
    )
);
