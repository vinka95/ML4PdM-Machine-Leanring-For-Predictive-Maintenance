import pytest

from ml4pdm.evaluation.metrics import loss_asymmetric, loss_false_negative_rate, loss_false_positive_rate, loss_lin_lin, score_performance

ERROR_STR = "[2, 1]"


def test_loss_lin_lin_1():
    result = loss_lin_lin([0.0], [0.0])
    expected = 0.0
    assert abs(result - expected) < 0.0000000001

    result = loss_lin_lin([1.2, 1.6], [1.5, 1.1])
    expected = 0.8
    assert abs(result - expected) < 0.0000000001

    result = loss_lin_lin([10.2, 5.999, 99.1, 5.1], [1.5, 10.116, 1.6, 10.6], 0.5, 0.75)
    expected = 84.4585
    assert abs(result - expected) < 0.0000000001


def test_loss_lin_lin_2():
    with pytest.raises(ValueError) as excinfo:
        loss_lin_lin([0.0, 5.0, 1.0], [0.0])
    assert "[3, 1]" in str(excinfo)


def test_loss_asymmetric_1():
    result = loss_asymmetric([0.0], [0.0])
    expected = 0.0
    assert abs(result - expected) < 0.0000000001

    result = loss_asymmetric([1.2, 1.1], [1.5, 1.6])
    expected = 0.0817256303
    assert abs(result - expected) < 0.0000000001

    result = loss_asymmetric([10.2, 5.999, 99.1, 5.1], [1.5, 10.116, 1.6, 10.6])
    expected = 1809.2377836670
    assert abs(result - expected) < 0.0000000001


def test_loss_asymmetric_2():
    with pytest.raises(ValueError) as excinfo:
        loss_asymmetric([0.0, 5.0], [0.0])
    assert ERROR_STR in str(excinfo)


def test_score_performance_1():
    result = score_performance([0.0], [0.0])
    expected = 1.0
    assert abs(result - expected) < 0.0000000001

    result = score_performance([-15.2, 1.1], [1.5, 1.6])
    expected = 0.5
    assert abs(result - expected) < 0.0000000001

    result = score_performance([19.2, 5.999, 14.6, 5.1], [1.5, 10.116, 1.6, 10.6])
    expected = 0.75
    assert abs(result - expected) < 0.0000000001


def test_score_performance_2():
    with pytest.raises(ValueError) as excinfo:
        score_performance([0.0, 5.0], [0.0])
    assert ERROR_STR in str(excinfo)


def test_loss_false_positive_rate_1():
    result = loss_false_positive_rate([0.0], [0.0])
    expected = 0.0
    assert abs(result - expected) < 0.0000000001

    result = loss_false_positive_rate([15.2, 1.1], [1.5, 1.6])
    expected = 0.5
    assert abs(result - expected) < 0.0000000001

    result = loss_false_positive_rate([19.2, 5.999, 14.6, 5.1], [1.5, 10.116, 1.6, 10.6])
    expected = 0.25
    assert abs(result - expected) < 0.0000000001


def test_loss_false_positive_rate_2():
    with pytest.raises(ValueError) as excinfo:
        loss_false_positive_rate([0.0, 5.0], [0.0])
    assert ERROR_STR in str(excinfo)


def test_loss_false_negative_rate_1():
    result = loss_false_negative_rate([0.0], [0.0])
    expected = 0.0
    assert abs(result - expected) < 0.0000000001

    result = loss_false_negative_rate([1.5, 1.1], [15.5, 1.6])
    expected = 0.5
    assert abs(result - expected) < 0.0000000001

    result = loss_false_negative_rate([19.2, 5.999, 17.6, 5.1], [1.5, 18.116, 1.6, 10.6])
    expected = 0.25
    assert abs(result - expected) < 0.0000000001


def test_loss_false_negative_rate_2():
    with pytest.raises(ValueError) as excinfo:
        loss_false_negative_rate([0.0, 5.0], [0.0])
    assert ERROR_STR in str(excinfo)
