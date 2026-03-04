#pragma once

#include "EnergyModel.hpp"
#include "MaxFlowSolver.hpp"
#include <memory>
#include <functional>

class AlphaExpansion {
public:
    using SolverFactory = std::function<std::unique_ptr<MaxFlowSolver>(int num_vars, int num_edges)>;

    AlphaExpansion(EnergyModel &model, SolverFactory solver_factory)
        : model_(model), solver_factory_(std::move(solver_factory)) {
    }

    std::unique_ptr<MaxFlowSolver> build_expansion_graph(const int alpha_label, const std::vector<int> &active_nodes,
                                                         std::vector<MaxFlowSolver::Var> &node_var_ids) const {
        const int num_active = active_nodes.size();
        if (num_active == 0) return nullptr;

        const int estimated_edges = num_active * 4;
        auto solver = solver_factory_(num_active, estimated_edges);

        node_var_ids.assign(model_.num_nodes(), -1);

        for (int i = 0; i < num_active; ++i) {
            int node = active_nodes[i];
            node_var_ids[node] = solver->add_variable();
        }

        for (const int node: active_nodes) {
            const MaxFlowSolver::Var var = node_var_ids[node];
            const int current_label = model_.get_label(node);
            const EnergyValue e0 = model_.get_unary_cost(node, alpha_label);
            const EnergyValue e1 = model_.get_unary_cost(node, current_label);
            solver->add_term1(var, e0, e1);
        }

        for (const int node_i: active_nodes) {
            const int current_label_i = model_.get_label(node_i);
            const MaxFlowSolver::Var var_i = node_var_ids[node_i];

            for (const int node_j: model_.get_neighbors(node_i)) {
                if (node_var_ids[node_j] != -1) {
                    if (node_i < node_j) {
                        const MaxFlowSolver::Var var_j = node_var_ids[node_j];
                        const int current_label_j = model_.get_label(node_j);
                        const EnergyValue e00 = model_.get_pairwise_cost(node_i, node_j, alpha_label, alpha_label);
                        const EnergyValue e01 = model_.get_pairwise_cost(node_i, node_j, alpha_label, current_label_j);
                        const EnergyValue e10 = model_.get_pairwise_cost(node_i, node_j, current_label_i, alpha_label);
                        const EnergyValue e11 = model_.get_pairwise_cost(node_i, node_j, current_label_i, current_label_j);
                        solver->add_term2(var_i, var_j, e00, e01, e10, e11);
                    }
                } else {
                    const int current_label_j = model_.get_label(node_j);
                    const EnergyValue e0 = model_.get_pairwise_cost(node_i, node_j, alpha_label, current_label_j);
                    const EnergyValue e1 = model_.get_pairwise_cost(node_i, node_j, current_label_i, current_label_j);
                    solver->add_term1(var_i, e0, e1);
                }
            }
        }

        return solver;
    }

    [[nodiscard]] bool perform_expansion_move(const int alpha_label) const {
        const std::vector<int> active_nodes = model_.get_active_nodes(alpha_label);
        if (active_nodes.empty()) return false;

        std::vector<MaxFlowSolver::Var> node_var_ids;
        const auto solver = build_expansion_graph(alpha_label, active_nodes, node_var_ids);

        solver->minimize();

        bool changed = false;
        std::vector<int> proposed_labels = model_.get_labels();

        for (const int node: active_nodes) {
            if (const MaxFlowSolver::Var var = node_var_ids[node]; solver->get_var(var) == 0) {
                proposed_labels[node] = alpha_label;
                changed = true;
            }
        }

        if (changed) {
            EnergyValue old_energy = model_.evaluate_total_energy();
            EnergyValue new_energy = model_.evaluate_total_energy(proposed_labels);
            if (new_energy < old_energy) {
                model_.set_labels(proposed_labels);
                return true;
            }
        }

        return false;
    }

private:
    EnergyModel &model_;
    SolverFactory solver_factory_;
};
