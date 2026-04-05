#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>

#include "core/EnergyModel.hpp"
#include "core/AlphaExpansion.hpp"
#include "strategies/SequentialStrategy.hpp"
#include "strategies/GreedyStrategy.hpp"
#include "strategies/RandomizedStrategy.hpp"
#include "solvers/BKSolver.hpp"
#ifdef USE_OR_TOOLS
#include "solvers/ORToolsSolver.hpp"
#endif

namespace py = pybind11;

template <typename T>
void bind_types(py::module &m, const std::string &type_suffix) {
    std::string em_name = "EnergyModel" + type_suffix;
    py::class_<EnergyModel<T>>(m, em_name.c_str(),
        "Stores the graph and energy costs for the Alpha-Expansion algorithm.\n\n"
        "Prefer the ``EnergyModel(num_nodes, num_labels, dtype)`` factory function over\n"
        "instantiating typed variants (``EnergyModelInt``, etc.) directly.")
        .def(py::init<int, int>(), py::arg("num_nodes"), py::arg("num_labels"),
             "Create an energy model with all labels set to 0.\n\n"
             ":param num_nodes: Number of nodes (pixels, graph vertices, ...).\n"
             ":param num_labels: Number of labels (classes, segments, communities, ...).")
        .def_property_readonly("num_nodes", &EnergyModel<T>::num_nodes,
             "Total number of nodes.")
        .def_property_readonly("num_labels", &EnergyModel<T>::num_labels,
             "Total number of labels.")
        .def("get_label", &EnergyModel<T>::get_label, py::arg("node"),
             "Return the current label assigned to *node*.")
        .def("set_label", &EnergyModel<T>::set_label, py::arg("node"), py::arg("label"),
             "Assign *label* to *node*.")
        .def("get_labels", &EnergyModel<T>::get_labels,
             "Return the full label list (one integer per node).")
        .def("set_labels", &EnergyModel<T>::set_labels, py::arg("labels"),
             "Replace the full label list. Length must equal ``num_nodes``.")
        .def("set_unary_cost_fn", &EnergyModel<T>::set_unary_cost_fn, py::arg("fn"),
             "Set a callback ``fn(node, label) -> cost`` for unary costs.\n"
             "If ``set_unary_costs()`` is also called, the dense array takes priority.")
        .def("set_pairwise_cost_fn", &EnergyModel<T>::set_pairwise_cost_fn, py::arg("fn"),
             "Set a callback ``fn(node1, node2, label1, label2) -> cost`` for pairwise costs.\n"
             "If ``set_pairwise_costs()`` or ``set_edge_weights()`` is also called, those take priority.")
        .def("set_unary_costs", &EnergyModel<T>::set_unary_costs, py::arg("costs"),
             "Set unary costs from a flat list of length ``num_nodes * num_labels``.\n"
             "Entry ``[node * num_labels + label]`` is the cost of assigning *label* to *node*.")
        .def("set_pairwise_costs", &EnergyModel<T>::set_pairwise_costs, py::arg("costs"),
             "Set a global pairwise cost matrix of length ``num_labels * num_labels``.\n"
             "Entry ``[l1 * num_labels + l2]`` is the cost of placing labels l1 and l2 on adjacent nodes.")
        .def("get_unary_cost", &EnergyModel<T>::get_unary_cost, py::arg("node"), py::arg("label"),
             "Return the unary cost for assigning *label* to *node*.")
        .def("get_pairwise_cost", &EnergyModel<T>::get_pairwise_cost,
             py::arg("node1"), py::arg("node2"), py::arg("label1"), py::arg("label2"),
             "Return the pairwise cost for the given node-label pair.")
        .def("set_edge_weights", &EnergyModel<T>::set_edge_weights,
             py::arg("n1s"), py::arg("n2s"), py::arg("weights"),
             "Set per-edge smoothness weights (Potts model).\n"
             "Cost is 0 when two adjacent nodes share the same label, *weight* otherwise.\n"
             "All three lists must have the same length.")
        .def("add_neighbor", &EnergyModel<T>::add_neighbor, py::arg("node1"), py::arg("node2"),
             "Add an undirected edge between *node1* and *node2*.")
        .def("add_grid_edges", &EnergyModel<T>::add_grid_edges, py::arg("width"), py::arg("height"),
             "Add a 4-connected grid neighborhood for an image of size *width* x *height*.\n"
             "Requires ``width * height == num_nodes``.")
        .def("get_neighbors", &EnergyModel<T>::get_neighbors, py::arg("node"),
             "Return the list of neighbors of *node*.")
        .def("get_active_nodes", &EnergyModel<T>::get_active_nodes, py::arg("alpha_label"),
             "Return indices of all nodes that do not currently have *alpha_label*.")
        .def("evaluate_total_energy",
             py::overload_cast<>(&EnergyModel<T>::evaluate_total_energy, py::const_),
             "Return the total energy for the current label assignment.")
        .def("evaluate_total_energy_with_labels",
             py::overload_cast<const std::vector<int>&>(&EnergyModel<T>::evaluate_total_energy, py::const_),
             py::arg("eval_labels"),
             "Return the total energy for the given label assignment.");

    std::string ae_name = "AlphaExpansion" + type_suffix;
    py::class_<AlphaExpansion<T>>(m, ae_name.c_str(),
        "Performs alpha-expansion moves on an EnergyModel.\n\n"
        "The model passed to the constructor must stay alive for the whole lifetime of this object.")
        .def(py::init([](EnergyModel<T>& model, const std::string& solver_type) {
            typename AlphaExpansion<T>::SolverFactory factory;
            if (solver_type == "bk") {
                factory = [](int v, int e) { return std::make_unique<BKSolver<T>>(v, e); };
            } else if (solver_type == "ortools") {
#ifdef USE_OR_TOOLS
                factory = [](int v, int e) { return std::make_unique<ORToolsSolver<T>>(); };
#else
                throw std::invalid_argument("OR-Tools solver not enabled at compile time.");
#endif
            } else {
                throw std::invalid_argument("Unknown solver type: " + solver_type + ". Use 'bk' or 'ortools'.");
            }
            return std::make_unique<AlphaExpansion<T>>(model, factory);
        }), py::arg("model"), py::arg("solver_type") = "bk",
        "Create an optimizer for *model*.\n\n"
        ":param model: The ``EnergyModel`` to optimize. Must outlive this object.\n"
        ":param solver_type: Max-flow backend -- ``'bk'`` (default) or ``'ortools'``.")
        .def("perform_expansion_move", &AlphaExpansion<T>::perform_expansion_move,
             py::arg("alpha_label"),
             "Attempt one alpha-expansion move for *alpha_label*.\n\n"
             "Returns ``True`` if any node changed its label and energy decreased.");

    std::string seq_name = "SequentialStrategy" + type_suffix;
    py::class_<SequentialStrategy<T>>(m, seq_name.c_str(),
        "Cycles through labels in fixed order 0, 1, ..., K-1 until convergence.")
        .def(py::init<int>(), py::arg("max_cycles") = 100,
             ":param max_cycles: Maximum number of full cycles (default: 100).")
        .def("execute", &SequentialStrategy<T>::execute, py::arg("optimizer"), py::arg("model"),
             "Run sequential alpha-expansion. Returns the number of cycles completed.");

    std::string greedy_name = "GreedyStrategy" + type_suffix;
    py::class_<GreedyStrategy<T>>(m, greedy_name.c_str(),
        "Each cycle picks the label whose expansion reduces the energy the most.")
        .def(py::init<int>(), py::arg("max_cycles") = 100,
             ":param max_cycles: Maximum number of greedy cycles (default: 100).")
        .def("execute", &GreedyStrategy<T>::execute, py::arg("optimizer"), py::arg("model"),
             "Run greedy alpha-expansion. Returns the number of cycles completed.");

    std::string rand_name = "RandomizedStrategy" + type_suffix;
    py::class_<RandomizedStrategy<T>>(m, rand_name.c_str(),
        "Shuffles the label order randomly at the start of each cycle.")
        .def(py::init<int, unsigned int>(), py::arg("max_cycles") = 100, py::arg("seed") = 42,
             ":param max_cycles: Maximum number of cycles (default: 100).\n"
             ":param seed: RNG seed for reproducibility (default: 42).")
        .def("execute", &RandomizedStrategy<T>::execute, py::arg("optimizer"), py::arg("model"),
             "Run randomized alpha-expansion. Returns the number of cycles completed.");
}

PYBIND11_MODULE(alpha_expansion_py, m) {
    m.doc() =
        "Python bindings for the Alpha Expansion library.\n\n"
        "Provides ``EnergyModel``, ``AlphaExpansion`` and three expansion strategies.\n"
        "Use the ``EnergyModel(num_nodes, num_labels, dtype)`` factory to create a model.\n\n"
        "Example::\n\n"
        "    import sys\n"
        "    sys.path.append('build')\n"
        "    import alpha_expansion_py as ae\n\n"
        "    model = ae.EnergyModel(10, 3, dtype='int32')\n"
        "    optimizer = ae.AlphaExpansionInt(model, 'bk')\n"
        "    strategy = ae.SequentialStrategyInt(max_cycles=100)\n"
        "    cycles = strategy.execute(optimizer, model)";

    bind_types<int32_t>(m, "Int");
    bind_types<float>(m, "Float");
    bind_types<double>(m, "Double");

    m.def("EnergyModel", [](int num_nodes, int num_labels, const std::string& dtype) -> py::object {
        if (dtype == "int32") return py::cast(EnergyModel<int32_t>(num_nodes, num_labels));
        if (dtype == "float32") return py::cast(EnergyModel<float>(num_nodes, num_labels));
        if (dtype == "float64") return py::cast(EnergyModel<double>(num_nodes, num_labels));
        throw std::invalid_argument("dtype must be 'int32', 'float32', or 'float64'");
    }, py::arg("num_nodes"), py::arg("num_labels"), py::arg("dtype") = "int32",
    "Create an EnergyModel of the given dtype.\n\n"
    ":param num_nodes: Number of nodes.\n"
    ":param num_labels: Number of labels.\n"
    ":param dtype: One of ``'int32'`` (default), ``'float32'``, ``'float64'``.");
}
