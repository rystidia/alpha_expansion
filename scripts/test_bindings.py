import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))

import alpha_expansion_py as ae


def test_two_pixel_trap():
    model = ae.EnergyModel(2, 3)
    model.set_label(0, 0)
    model.set_label(1, 0)

    def unary_cost(node, label):
        if node == 0:
            return 1 if label == 0 else (3 if label == 1 else 0)
        return 1 if label == 0 else (0 if label == 1 else 3)

    def pairwise_cost(n1, n2, l1, l2):
        return 0 if l1 == l2 else 1

    model.set_unary_cost_fn(unary_cost)
    model.set_pairwise_cost_fn(pairwise_cost)
    model.add_neighbor(0, 1)

    print(f"Initial energy: {model.evaluate_total_energy()}")

    # Try BK Sequential
    opt = ae.AlphaExpansion(model, "bk")
    strategy = ae.SequentialStrategy(100)
    strategy.execute(opt, model)

    final_energy = model.evaluate_total_energy()
    print(f"Final energy (BK Sequential): {final_energy}")
    assert final_energy == 2, f"Expected 2, got {final_energy}"

    # Try OR-Tools Sequential
    model.set_label(0, 0)
    model.set_label(1, 0)

    opt_or_seq = ae.AlphaExpansion(model, "ortools")
    strategy_or_seq = ae.SequentialStrategy(100)
    strategy_or_seq.execute(opt_or_seq, model)

    final_energy_or_seq = model.evaluate_total_energy()
    print(f"Final energy (ORTools Sequential): {final_energy_or_seq}")
    assert final_energy_or_seq == 2, f"Expected 2, got {final_energy_or_seq}"

    # Try BK Greedy from a solvable starting position
    model.set_label(0, 0)
    model.set_label(1, 1)

    opt_greedy = ae.AlphaExpansion(model, "bk")
    strategy_greedy = ae.GreedyStrategy(100)
    strategy_greedy.execute(opt_greedy, model)

    final_energy_greedy = model.evaluate_total_energy()
    print(f"Final energy (BK Greedy): {final_energy_greedy}")
    assert final_energy_greedy == 1, f"Expected 1, got {final_energy_greedy}"

    # Try OR-Tools Greedy
    model.set_label(0, 0)
    model.set_label(1, 1)

    opt_greedy_or = ae.AlphaExpansion(model, "ortools")
    strategy_greedy_or = ae.GreedyStrategy(100)
    strategy_greedy_or.execute(opt_greedy_or, model)

    final_energy_greedy_or = model.evaluate_total_energy()
    print(f"Final energy (ORTools Greedy): {final_energy_greedy_or}")
    assert final_energy_greedy_or == 1, f"Expected 1, got {final_energy_greedy_or}"

    print("Python bindings test passed!")


if __name__ == "__main__":
    test_two_pixel_trap()
