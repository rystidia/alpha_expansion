#pragma once

#include <vector>
#include <functional>
#include <unordered_map>
#include <cstdint>

/// @brief Stores the graph and energy costs for the Alpha-Expansion algorithm.
///
/// An EnergyModel holds the nodes, the neighbor graph and the unary and pairwise cost
/// functions that define the energy to be minimized. Costs can be set as dense arrays
/// (good for image grids) or as callback functions (good for general graphs).
/// If both are set, the array is used.
///
/// @tparam T Numeric type for costs (`int32_t`, `float`, or `double`).
template <typename T>
class EnergyModel {
public:
    /// Callable that returns the unary cost for assigning @p label to @p node.
    using UnaryCostFn = std::function<T(int node, int label)>;
    /// Callable that returns the pairwise cost for assigning @p label1 to @p node1
    /// and @p label2 to @p node2.
    using PairwiseCostFn = std::function<T(int node1, int node2, int label1, int label2)>;

    /// @brief Constructs an energy model with all labels initialized to 0.
    /// @param num_nodes  Number of nodes (pixels, graph vertices, etc.).
    /// @param num_labels Number of labels (classes, segments, communities, etc.).
    EnergyModel(int num_nodes, int num_labels)
        : num_nodes_(num_nodes), num_labels_(num_labels), labels_(num_nodes, 0), neighbors_(num_nodes) {}

    /// @brief Returns the total number of nodes.
    [[nodiscard]] int num_nodes() const { return num_nodes_; }

    /// @brief Returns the total number of labels.
    [[nodiscard]] int num_labels() const { return num_labels_; }

    /// @brief Returns the current label assigned to @p node.
    [[nodiscard]] int get_label(int node) const { return labels_[node]; }

    /// @brief Assigns @p label to @p node.
    void set_label(int node, int label) { labels_[node] = label; }

    /// @brief Returns the full label vector (one entry per node).
    [[nodiscard]] const std::vector<int>& get_labels() const { return labels_; }

    /// @brief Replaces the full label vector.
    /// @param labels Must have size equal to `num_nodes()`.
    void set_labels(const std::vector<int>& labels) { labels_ = labels; }

    /// @brief Sets a callback function for unary costs.
    ///
    /// If `set_unary_costs()` is also called, the dense array takes priority over this callback.
    void set_unary_cost_fn(UnaryCostFn fn) { unary_cost_fn_ = fn; }

    /// @brief Sets a callback function for pairwise costs.
    ///
    /// If `set_pairwise_costs()` or `set_edge_weights()` is also called, those take priority.
    void set_pairwise_cost_fn(PairwiseCostFn fn) { pairwise_cost_fn_ = fn; }

    /// @brief Returns the unary cost for assigning @p label to @p node.
    ///
    /// Checks the dense array first. If no array is set, it uses the callback function.
    [[nodiscard]] T get_unary_cost(int node, int label) const {
        if (!unary_costs_.empty()) {
            return unary_costs_[node * num_labels_ + label];
        }
        return unary_cost_fn_ ? unary_cost_fn_(node, label) : 0;
    }

    /// @brief Returns the pairwise cost for the given node–label pair.
    ///
    /// Priority: per-edge weights > dense pairwise array > callback.
    [[nodiscard]] T get_pairwise_cost(int node1, int node2, int label1, int label2) const {
        if (!edge_weights_.empty()) {
            auto it = edge_weights_.find(edge_key(node1, node2));
            if (it != edge_weights_.end()) {
                return label1 == label2 ? T{0} : it->second;
            }
        }
        if (!pairwise_costs_.empty()) {
            return pairwise_costs_[label1 * num_labels_ + label2];
        }
        return pairwise_cost_fn_ ? pairwise_cost_fn_(node1, node2, label1, label2) : 0;
    }

    /// @brief Sets unary costs from a flat row-major array of size `num_nodes * num_labels`.
    ///
    /// Entry `[node * num_labels + label]` is the cost of assigning @p label to @p node.
    /// This overrides any callback set via `set_unary_cost_fn()`.
    /// @throws std::invalid_argument if the array size does not match.
    void set_unary_costs(const std::vector<T>& costs) {
        if (costs.size() != static_cast<size_t>(num_nodes_ * num_labels_)) {
            throw std::invalid_argument("Unary costs array must have size num_nodes * num_labels");
        }
        unary_costs_ = costs;
    }

    /// @brief Sets a global pairwise cost matrix of size `num_labels * num_labels`.
    ///
    /// Entry `[l1 * num_labels + l2]` is the cost of placing label l1 and l2 on adjacent nodes.
    /// This overrides any callback, but per-edge weights set via `set_edge_weights()` take
    /// priority over this matrix.
    /// @throws std::invalid_argument if the array size does not match.
    void set_pairwise_costs(const std::vector<T>& costs) {
        if (costs.size() != static_cast<size_t>(num_labels_ * num_labels_)) {
            throw std::invalid_argument("Pairwise costs array must have size num_labels * num_labels");
        }
        pairwise_costs_ = costs;
    }

    /// @brief Sets per-edge smoothness weights (Potts model).
    ///
    /// For each edge (n1, n2), the pairwise cost is 0 if the two nodes have the same label,
    /// and @p weight if they have different labels. This overrides both the dense pairwise
    /// array and the callback for the specified edges.
    /// @throws std::invalid_argument if the three vectors have different sizes.
    void set_edge_weights(const std::vector<int>& n1s, const std::vector<int>& n2s, const std::vector<T>& weights) {
        if (n1s.size() != n2s.size() || n1s.size() != weights.size()) {
            throw std::invalid_argument("n1s, n2s, and weights must have the same size");
        }
        for (size_t i = 0; i < n1s.size(); ++i) {
            edge_weights_[edge_key(n1s[i], n2s[i])] = weights[i];
        }
    }

    /// @brief Adds an undirected edge between @p node1 and @p node2.
    void add_neighbor(int node1, int node2) {
        neighbors_[node1].push_back(node2);
        neighbors_[node2].push_back(node1);
    }

    /// @brief Populates a 4-connected grid neighbourhood for an image of size @p width × @p height.
    /// @throws std::invalid_argument if `width * height != num_nodes()`.
    void add_grid_edges(int width, int height) {
        if (width * height != num_nodes_) {
            throw std::invalid_argument("Grid dimensions do not match the number of nodes");
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int node = y * width + x;
                if (x + 1 < width)  add_neighbor(node, node + 1);
                if (y + 1 < height) add_neighbor(node, node + width);
            }
        }
    }

    /// @brief Returns the neighbours of @p node.
    [[nodiscard]] const std::vector<int>& get_neighbors(int node) const {
        return neighbors_[node];
    }

    /// @brief Returns the indices of all nodes that do not currently have @p alpha_label.
    ///
    /// These are the nodes that take part in an alpha-expansion move for @p alpha_label.
    [[nodiscard]] std::vector<int> get_active_nodes(int alpha_label) const {
        std::vector<int> active_nodes;
        active_nodes.reserve(num_nodes_);
        for (int i = 0; i < num_nodes_; ++i) {
            if (labels_[i] != alpha_label) active_nodes.push_back(i);
        }
        return active_nodes;
    }

    /// @brief Evaluates the total energy for a given label assignment.
    /// @param eval_labels Label vector of length `num_nodes()`.
    [[nodiscard]] T evaluate_total_energy(const std::vector<int> &eval_labels) const {
        T total = 0;
        for (int i = 0; i < num_nodes_; ++i) {
            total += get_unary_cost(i, eval_labels[i]);
            for (int neighbor : neighbors_[i]) {
                if (i < neighbor) {
                    total += get_pairwise_cost(i, neighbor, eval_labels[i], eval_labels[neighbor]);
                }
            }
        }
        return total;
    }

    /// @brief Evaluates the total energy for the model's current label assignment.
    [[nodiscard]] T evaluate_total_energy() const {
        return evaluate_total_energy(labels_);
    }

private:
    int64_t edge_key(int n1, int n2) const {
        int a = std::min(n1, n2), b = std::max(n1, n2);
        return (int64_t)a * num_nodes_ + b;
    }

    int num_nodes_;
    int num_labels_;
    std::vector<int> labels_;
    std::vector<std::vector<int>> neighbors_;
    UnaryCostFn unary_cost_fn_;
    PairwiseCostFn pairwise_cost_fn_;
    std::vector<T> unary_costs_;
    std::vector<T> pairwise_costs_;
    std::unordered_map<int64_t, T> edge_weights_;
};
