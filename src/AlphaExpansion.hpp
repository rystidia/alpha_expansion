#pragma once

#include "EnergyModel.hpp"
#include "MaxFlowSolver.hpp"
#include <memory>
#include <functional>

class AlphaExpansion {
public:
    using SolverFactory = std::function<std::unique_ptr<MaxFlowSolver>(int num_vars, int num_edges)>;

    AlphaExpansion(EnergyModel& model, SolverFactory solver_factory)
        : model_(model), solver_factory_(std::move(solver_factory)) {}

    std::unique_ptr<MaxFlowSolver> build_expansion_graph(int alpha_label, const std::vector<int>& active_nodes, std::vector<MaxFlowSolver::Var>& node_var_ids) {
        int num_active = active_nodes.size();
        if (num_active == 0) return nullptr;

        int estimated_edges = num_active * 4;
        auto solver = solver_factory_(num_active, estimated_edges);

        node_var_ids.assign(model_.num_nodes(), -1);

        for (int i = 0; i < num_active; ++i) {
            int node = active_nodes[i];
            node_var_ids[node] = solver->add_variable();
        }

        for (int node : active_nodes) {
            MaxFlowSolver::Var var = node_var_ids[node];
            int current_label = model_.get_label(node);
            EnergyValue e0 = model_.get_unary_cost(node, alpha_label);
            EnergyValue e1 = model_.get_unary_cost(node, current_label);
            solver->add_term1(var, e0, e1);
        }

        for (int node_i : active_nodes) {
            int current_label_i = model_.get_label(node_i);
            MaxFlowSolver::Var var_i = node_var_ids[node_i];

            for (int node_j : model_.get_neighbors(node_i)) {
                if (node_var_ids[node_j] != -1) {
                    if (node_i < node_j) {
                        MaxFlowSolver::Var var_j = node_var_ids[node_j];
                        int current_label_j = model_.get_label(node_j);
                        EnergyValue e00 = model_.get_pairwise_cost(node_i, node_j, alpha_label, alpha_label);
                        EnergyValue e01 = model_.get_pairwise_cost(node_i, node_j, alpha_label, current_label_j);
                        EnergyValue e10 = model_.get_pairwise_cost(node_i, node_j, current_label_i, alpha_label);
                        EnergyValue e11 = model_.get_pairwise_cost(node_i, node_j, current_label_i, current_label_j);
                        solver->add_term2(var_i, var_j, e00, e01, e10, e11);
                    }
                } else {
                    int current_label_j = model_.get_label(node_j);
                    EnergyValue e0 = model_.get_pairwise_cost(node_i, node_j, alpha_label, current_label_j);
                    EnergyValue e1 = model_.get_pairwise_cost(node_i, node_j, current_label_i, current_label_j);
                    solver->add_term1(var_i, e0, e1);
                }
            }
        }

        return solver;
    }

    bool perform_expansion_move(int alpha_label) {
        std::vector<int> active_nodes = model_.get_active_nodes(alpha_label);
        if (active_nodes.empty()) return false;

        std::vector<MaxFlowSolver::Var> node_var_ids;
        auto solver = build_expansion_graph(alpha_label, active_nodes, node_var_ids);

        solver->minimize();

        bool changed = false;
        for (int node : active_nodes) {
            MaxFlowSolver::Var var = node_var_ids[node];
            if (solver->get_var(var) == 0) {
                model_.set_label(node, alpha_label);
                changed = true;
            }
        }
        
        return changed;
    }

private:
    EnergyModel& model_;
    SolverFactory solver_factory_;
};
