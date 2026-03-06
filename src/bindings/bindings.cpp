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
#include "solvers/ORToolsSolver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(alpha_expansion_py, m) {
    m.doc() = "Alpha Expansion Python Bindings";

    py::class_<EnergyModel>(m, "EnergyModel")
        .def(py::init<int, int>(), py::arg("num_nodes"), py::arg("num_labels"))
        .def_property_readonly("num_nodes", &EnergyModel::num_nodes)
        .def_property_readonly("num_labels", &EnergyModel::num_labels)
        .def("get_label", &EnergyModel::get_label, py::arg("node"))
        .def("set_label", &EnergyModel::set_label, py::arg("node"), py::arg("label"))
        .def("get_labels", &EnergyModel::get_labels)
        .def("set_labels", &EnergyModel::set_labels, py::arg("labels"))
        .def("set_unary_cost_fn", &EnergyModel::set_unary_cost_fn, py::arg("fn"))
        .def("set_pairwise_cost_fn", &EnergyModel::set_pairwise_cost_fn, py::arg("fn"))
        .def("set_unary_costs", &EnergyModel::set_unary_costs, py::arg("costs"))
        .def("set_pairwise_costs", &EnergyModel::set_pairwise_costs, py::arg("costs"))
        .def("get_unary_cost", &EnergyModel::get_unary_cost, py::arg("node"), py::arg("label"))
        .def("get_pairwise_cost", &EnergyModel::get_pairwise_cost, py::arg("node1"), py::arg("node2"), py::arg("label1"), py::arg("label2"))
        .def("add_neighbor", &EnergyModel::add_neighbor, py::arg("node1"), py::arg("node2"))
        .def("get_neighbors", &EnergyModel::get_neighbors, py::arg("node"))
        .def("get_active_nodes", &EnergyModel::get_active_nodes, py::arg("alpha_label"))
        .def("evaluate_total_energy", py::overload_cast<>(&EnergyModel::evaluate_total_energy, py::const_))
        .def("evaluate_total_energy_with_labels", py::overload_cast<const std::vector<int>&>(&EnergyModel::evaluate_total_energy, py::const_), py::arg("eval_labels"));

    py::class_<AlphaExpansion>(m, "AlphaExpansion")
        .def(py::init([](EnergyModel& model, const std::string& solver_type) {
            AlphaExpansion::SolverFactory factory;
            if (solver_type == "bk") {
                factory = [](int v, int e) { return std::make_unique<BKSolver>(v, e); };
            } else if (solver_type == "ortools") {
                factory = [](int v, int e) { return std::make_unique<ORToolsSolver>(); };
            } else {
                throw std::invalid_argument("Unknown solver type: " + solver_type + ". Use 'bk' or 'ortools'.");
            }
            return std::make_unique<AlphaExpansion>(model, factory);
        }), py::arg("model"), py::arg("solver_type") = "bk")
        .def("perform_expansion_move", &AlphaExpansion::perform_expansion_move, py::arg("alpha_label"));

    py::class_<SequentialStrategy>(m, "SequentialStrategy")
        .def(py::init<int>(), py::arg("max_cycles") = 100)
        .def("execute", &SequentialStrategy::execute, py::arg("optimizer"), py::arg("model"));

    py::class_<GreedyStrategy>(m, "GreedyStrategy")
        .def(py::init<int>(), py::arg("max_cycles") = 100)
        .def("execute", &GreedyStrategy::execute, py::arg("optimizer"), py::arg("model"));

    py::class_<RandomizedStrategy>(m, "RandomizedStrategy")
        .def(py::init<int, unsigned int>(), py::arg("max_cycles") = 100, py::arg("seed") = 42)
        .def("execute", &RandomizedStrategy::execute, py::arg("optimizer"), py::arg("model"));
}
