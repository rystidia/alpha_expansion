import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build")))

from experiments import build_chain, build_checkerboard, build_snake


def test_chain_size_matches():
    model, optimum = build_chain(n=7)
    assert model.num_nodes == 7
    assert model.num_labels == 3
    assert len(optimum) == 7

def test_chain_optimum_energy_smaller_than_uniform():
    model, optimum = build_chain(n=11)
    uniform = [0] * 11
    e_uniform = model.evaluate_total_energy_with_labels(uniform)
    e_opt = model.evaluate_total_energy_with_labels(optimum)
    assert e_opt < e_uniform

def test_checkerboard_size_matches():
    model, optimum = build_checkerboard(side=4)
    assert model.num_nodes == 16
    assert model.num_labels == 3

def test_snake_size_matches():
    model, optimum = build_snake(side=4)
    assert model.num_nodes == 16
