import numpy as np
from sklearn.model_selection import KFold

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.evaluation import Evaluator


def generate_dataset_mock(value=420):
    dataset = Dataset()
    dataset.data = np.asarray([[[(1, 1), (2, 4), (3, 8)]], [[(1, 1), (2, 4), (3, 8)]], [[(1, 1), (2, 4), (3, 8)]]])
    dataset.target = np.asarray([value, value, value])
    dataset.features = [('ts1', TIMESERIES(NUMERIC(), NUMERIC())), ('ts2', TIMESERIES(
        NUMERIC(), NUMERIC())), ('ts3', TIMESERIES(NUMERIC(), NUMERIC()))]
    return dataset


class PipelineMock():
    def __init__(self, value=69):
        self._value = value

    def fit(self, data: Dataset, y=None):
        return self

    def predict(self, data: Dataset):
        ret = np.zeros(len(data.data))
        ret.fill(self._value)
        return ret


def metric_maxpred_mock(y_true, y_pred):
    return np.max(y_pred)


def metric_maxtrue_mock(y_true, y_pred):
    return np.max(y_true)


def test_evaluator():
    # TEST 4 datasets, 3 pipelines and 2 metrics

    # 4 datasets
    dats = []
    dats.append(generate_dataset_mock(1))
    dats.append(generate_dataset_mock(2))
    dats.append(generate_dataset_mock(4))
    dats.append(generate_dataset_mock(8))

    # 3 pipelines
    pipes = []
    pipes.append(PipelineMock(420))
    pipes.append(PipelineMock(1337))
    pipes.append(PipelineMock(69))

    # 1 dataset_splitter
    splitter = KFold(n_splits=2, shuffle=False)

    # 2 evaluation metrics
    metrics = []
    metrics.append(metric_maxpred_mock)
    metrics.append(metric_maxtrue_mock)

    # Construct evaluator and execute evaluation method
    evaluator = Evaluator(datasets=dats, pipelines=pipes, dataset_splitter=splitter, evaluation_metrics=metrics)
    result = evaluator.evaluate()

    # Test shape of result
    assert result.shape == (4, 3, 2)

    # Test some values of result
    assert result[0, 0, 0] == 420
    assert result[0, 1, 0] == 1337
    assert result[0, 2, 0] == 69
    assert result[0, 0, 1] == 1
    assert result[1, 0, 1] == 2
    assert result[2, 1, 1] == 4
    assert result[3, 2, 1] == 8
