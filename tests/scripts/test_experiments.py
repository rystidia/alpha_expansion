import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build")))

import numpy as np
from experiments import build_chain, build_checkerboard, build_snake, build_restoration_model, build_stereo_model


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


def test_build_restoration_model_small():
    noisy = np.full((4, 4), 128, dtype=np.uint8)
    model, levels = build_restoration_model(noisy, num_labels=4, lambda_smooth=10)
    assert model.num_nodes == 16
    assert model.num_labels == 4
    assert len(levels) == 4


def test_build_stereo_model_small():
    left = np.zeros((4, 8, 3), dtype=np.float32)
    right = np.zeros((4, 8, 3), dtype=np.float32)
    model = build_stereo_model(left, right, num_labels=4)
    assert model.num_nodes == 32
    assert model.num_labels == 4
