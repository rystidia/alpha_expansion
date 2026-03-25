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
    py::class_<EnergyModel<T>>(m, em_name.c_str())
        .def(py::init<int, int>(), py::arg("num_nodes"), py::arg("num_labels"))
        .def_property_readonly("num_nodes", &EnergyModel<T>::num_nodes)
        .def_property_readonly("num_labels", &EnergyModel<T>::num_labels)
        .def("get_label", &EnergyModel<T>::get_label, py::arg("node"))
        .def("set_label", &EnergyModel<T>::set_label, py::arg("node"), py::arg("label"))
        .def("get_labels", &EnergyModel<T>::get_labels)
        .def("set_labels", &EnergyModel<T>::set_labels, py::arg("labels"))
        .def("set_unary_cost_fn", &EnergyModel<T>::set_unary_cost_fn, py::arg("fn"))
        .def("set_pairwise_cost_fn", &EnergyModel<T>::set_pairwise_cost_fn, py::arg("fn"))
        .def("set_unary_costs", &EnergyModel<T>::set_unary_costs, py::arg("costs"))
        .def("set_pairwise_costs", &EnergyModel<T>::set_pairwise_costs, py::arg("costs"))
        .def("get_unary_cost", &EnergyModel<T>::get_unary_cost, py::arg("node"), py::arg("label"))
        .def("get_pairwise_cost", &EnergyModel<T>::get_pairwise_cost, py::arg("node1"), py::arg("node2"), py::arg("label1"), py::arg("label2"))
        .def("add_neighbor", &EnergyModel<T>::add_neighbor, py::arg("node1"), py::arg("node2"))
        .def("add_grid_edges", &EnergyModel<T>::add_grid_edges, py::arg("width"), py::arg("height"))
        .def("get_neighbors", &EnergyModel<T>::get_neighbors, py::arg("node"))
        .def("get_active_nodes", &EnergyModel<T>::get_active_nodes, py::arg("alpha_label"))
        .def("evaluate_total_energy", py::overload_cast<>(&EnergyModel<T>::evaluate_total_energy, py::const_))
        .def("evaluate_total_energy_with_labels", py::overload_cast<const std::vector<int>&>(&EnergyModel<T>::evaluate_total_energy, py::const_), py::arg("eval_labels"));

    std::string ae_name = "AlphaExpansion" + type_suffix;
    py::class_<AlphaExpansion<T>>(m, ae_name.c_str())
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
        }), py::arg("model"), py::arg("solver_type") = "bk")
        .def("perform_expansion_move", &AlphaExpansion<T>::perform_expansion_move, py::arg("alpha_label"));

    std::string seq_name = "SequentialStrategy" + type_suffix;
    py::class_<SequentialStrategy<T>>(m, seq_name.c_str())
        .def(py::init<int>(), py::arg("max_cycles") = 100)
        .def("execute", &SequentialStrategy<T>::execute, py::arg("optimizer"), py::arg("model"));

    std::string greedy_name = "GreedyStrategy" + type_suffix;
    py::class_<GreedyStrategy<T>>(m, greedy_name.c_str())
        .def(py::init<int>(), py::arg("max_cycles") = 100)
        .def("execute", &GreedyStrategy<T>::execute, py::arg("optimizer"), py::arg("model"));

    std::string rand_name = "RandomizedStrategy" + type_suffix;
    py::class_<RandomizedStrategy<T>>(m, rand_name.c_str())
        .def(py::init<int, unsigned int>(), py::arg("max_cycles") = 100, py::arg("seed") = 42)
        .def("execute", &RandomizedStrategy<T>::execute, py::arg("optimizer"), py::arg("model"));
}

PYBIND11_MODULE(alpha_expansion_py, m) {
    m.doc() = "Alpha Expansion Python Bindings";

    bind_types<int32_t>(m, "Int");
    bind_types<float>(m, "Float");
    bind_types<double>(m, "Double");

    // Add Python factory functions for ease of use
    m.def("EnergyModel", [](int num_nodes, int num_labels, const std::string& dtype) -> py::object {
        if (dtype == "int32") return py::cast(EnergyModel<int32_t>(num_nodes, num_labels));
        if (dtype == "float32") return py::cast(EnergyModel<float>(num_nodes, num_labels));
        if (dtype == "float64") return py::cast(EnergyModel<double>(num_nodes, num_labels));
        throw std::invalid_argument("dtype must be 'int32', 'float32', or 'float64'");
    }, py::arg("num_nodes"), py::arg("num_labels"), py::arg("dtype") = "int32",
    "Creates an EnergyModel of the specified dtype ('int32', 'float32', 'float64')");
}
