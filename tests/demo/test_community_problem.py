import pytest

from demo.interactive_segmentation.community import CommunityDetectionProblem


def test_build_model_raises_without_selection():
    problem = CommunityDetectionProblem()
    with pytest.raises(ValueError, match="No graph selected"):
        problem.build_model()


def test_num_labels_without_selection():
    problem = CommunityDetectionProblem()
    assert problem.num_labels() == 0


def test_karate_build_model():
    problem = CommunityDetectionProblem()
    problem._select_graph(0)
    model, optimizer = problem.build_model()
    assert model is not None
    assert optimizer is not None
    assert problem.num_labels() == 2


def test_lesmis_build_model():
    problem = CommunityDetectionProblem()
    problem._select_graph(1)
    model, optimizer = problem.build_model()
    assert model is not None
    assert optimizer is not None
    assert problem.num_labels() == 3


def test_karate_convergence():
    import alpha_expansion_py as ae

    problem = CommunityDetectionProblem()
    problem._select_graph(0)
    model, optimizer = problem.build_model()
    strategy = ae.SequentialStrategyInt(50)
    cycles = strategy.execute(optimizer, model)
    assert cycles >= 1
    labels = model.get_labels()
    assert len(labels) == 34
    assert all(0 <= lbl < 2 for lbl in labels)
