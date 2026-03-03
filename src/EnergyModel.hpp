#pragma once

#include <vector>
#include <functional>
#include <cstdint>

typedef int32_t EnergyValue;

class EnergyModel {
public:
    using UnaryCostFn = std::function<EnergyValue(int node, int label)>;
    using PairwiseCostFn = std::function<EnergyValue(int node1, int node2, int label1, int label2)>;

    EnergyModel(int num_nodes, int num_labels)
        : num_nodes_(num_nodes), num_labels_(num_labels), labels_(num_nodes, 0), neighbors_(num_nodes) {}

    int num_nodes() const { return num_nodes_; }
    int num_labels() const { return num_labels_; }

    int get_label(int node) const { return labels_[node]; }
    void set_label(int node, int label) { labels_[node] = label; }
    const std::vector<int>& get_labels() const { return labels_; }

    void set_unary_cost_fn(UnaryCostFn fn) { unary_cost_fn_ = fn; }
    void set_pairwise_cost_fn(PairwiseCostFn fn) { pairwise_cost_fn_ = fn; }

    EnergyValue get_unary_cost(int node, int label) const {
        return unary_cost_fn_ ? unary_cost_fn_(node, label) : 0;
    }

    EnergyValue get_pairwise_cost(int node1, int node2, int label1, int label2) const {
        return pairwise_cost_fn_ ? pairwise_cost_fn_(node1, node2, label1, label2) : 0;
    }

    void add_neighbor(int node1, int node2) {
        neighbors_[node1].push_back(node2);
        neighbors_[node2].push_back(node1);
    }

    const std::vector<int>& get_neighbors(int node) const {
        return neighbors_[node];
    }
    
    // returns a list of indices with label different from alpha_label
    std::vector<int> get_active_nodes(int alpha_label) const {
        std::vector<int> active_nodes;
        active_nodes.reserve(num_nodes_);
        for (int i = 0; i < num_nodes_; ++i) {
            if (labels_[i] != alpha_label) {
                active_nodes.push_back(i);
            }
        }
        return active_nodes;
    }
    
    EnergyValue evaluate_total_energy() const {
        EnergyValue total = 0;
        for (int i = 0; i < num_nodes_; ++i) {
            total += get_unary_cost(i, labels_[i]);
            for (int neighbor : neighbors_[i]) {
                if (i < neighbor) {
                    total += get_pairwise_cost(i, neighbor, labels_[i], labels_[neighbor]);
                }
            }
        }
        return total;
    }

private:
    int num_nodes_;
    int num_labels_;
    std::vector<int> labels_;
    std::vector<std::vector<int>> neighbors_;
    UnaryCostFn unary_cost_fn_;
    PairwiseCostFn pairwise_cost_fn_;
};
