#include <gtest/gtest.h>
#include "energy.h"
#include <vector>

typedef Energy<int, int, int> EnergyT;

TEST(EnergyMinimization, BasicSubmodularEnergy) {
    EnergyT *e = new EnergyT(3, 5);
    EnergyT::Var varx = e->add_variable();
    EnergyT::Var vary = e->add_variable();
    EnergyT::Var varz = e->add_variable();

    e->add_term1(varx, 0, 1);
    e->add_term1(vary, 0, -2);
    e->add_term1(varz, 3, 0);
    e->add_term2(varx, vary, 0, 0, 0, -4);
    e->add_term2(vary, varz, 0, 5, 5, 0); 

    EXPECT_EQ(e->minimize(), -5);
    EXPECT_EQ(e->get_var(varx), 1);
    EXPECT_EQ(e->get_var(vary), 1);
    EXPECT_EQ(e->get_var(varz), 1);
    delete e;
}

TEST(EnergyMinimization, IndependentVariables) {
    EnergyT *e = new EnergyT(5, 0);
    std::vector<EnergyT::Var> vars;
    for (int i = 0; i < 5; ++i) {
        vars.push_back(e->add_variable());
    }

    e->add_term1(vars[0], 10, 0); 
    e->add_term1(vars[1], 0, 10); 
    e->add_term1(vars[2], -5, 5); 
    e->add_term1(vars[3], 5, -5); 
    e->add_term1(vars[4], 100, 100); 

    EXPECT_EQ(e->minimize(), -5 - 5 + 100);
    EXPECT_EQ(e->get_var(vars[0]), 1);
    EXPECT_EQ(e->get_var(vars[1]), 0);
    EXPECT_EQ(e->get_var(vars[2]), 0);
    EXPECT_EQ(e->get_var(vars[3]), 1);
    delete e;
}

TEST(EnergyMinimization, AttractivePotentials) {
    EnergyT *e = new EnergyT(2, 1);
    EnergyT::Var x = e->add_variable();
    EnergyT::Var y = e->add_variable();

    e->add_term1(x, 100, 0); 
    e->add_term1(y, 0, 10);  

    e->add_term2(x, y, 0, 1000, 1000, 0);

    EXPECT_EQ(e->minimize(), 10);
    EXPECT_EQ(e->get_var(x), 1);
    EXPECT_EQ(e->get_var(y), 1);
    delete e;
}

TEST(EnergyMinimization, 1DChainMRF) {
    const int N = 10;
    EnergyT *e = new EnergyT(N, N - 1);
    std::vector<EnergyT::Var> vars(N);
    for (int i = 0; i < N; ++i) {
        vars[i] = e->add_variable();
    }

    e->add_term1(vars[0], 1000, 0); 
    e->add_term1(vars[N - 1], 0, 1000); 

    for (int i = 0; i < N - 1; ++i) {
        e->add_term2(vars[i], vars[i+1], 0, 1, 1, 0);
    }

    EXPECT_EQ(e->minimize(), 1);
    EXPECT_EQ(e->get_var(vars[0]), 1);
    EXPECT_EQ(e->get_var(vars[N - 1]), 0);
    delete e;
}

TEST(EnergyMinimization, 2DGridMRF) {
    const int W = 3;
    const int H = 3;
    EnergyT *e = new EnergyT(W * H, 4 * W * H);
    std::vector<EnergyT::Var> vars(W * H);
    for (int i = 0; i < W * H; ++i) {
        vars[i] = e->add_variable();
    }

    e->add_term1(vars[0], 100, 0);
    e->add_term1(vars[8], 0, 100);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int i = y * W + x;
            if (x < W - 1) e->add_term2(vars[i], vars[i + 1], 0, 1, 1, 0); // right
            if (y < H - 1) e->add_term2(vars[i], vars[i + W], 0, 1, 1, 0); // down
        }
    }

    EXPECT_EQ(e->minimize(), 2);
    delete e;
}
