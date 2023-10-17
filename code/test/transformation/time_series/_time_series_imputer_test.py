from test.transformation.base_test import generate_mock_dataset_wmissing
from typing import Tuple

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.transformation import TimeSeriesImputer


def generate_weird_timestep_timeseries_dataset():
    dataset = Dataset()
    dataset.data = [[[(0, 1), (2, 5), (4, 0), (7, 4)]], [[(1, 4), (3, 5), (5, 9)]]]
    dataset.features = [('ts1', TIMESERIES(NUMERIC(), NUMERIC()))]
    return dataset


FEATURE_NAME = 'imputed(ts1)'


def check_4_strategies_on(dataset, resulting_timesteps):
    check_strategy(dataset, "mean-by-step", resulting_timesteps)
    check_strategy(dataset, "full-mean", resulting_timesteps)
    check_strategy(dataset, "interpolation-linear", resulting_timesteps)
    check_strategy(dataset, "interpolation-quadratic", resulting_timesteps)


def check_strategy(dataset, strategy_string, resulting_timesteps):
    imputer = TimeSeriesImputer(fill_strategy=strategy_string)
    t_dataset = imputer.fit_transform(dataset, None)
    assert isinstance(t_dataset, Dataset)
    assert t_dataset.features[0][0] == FEATURE_NAME
    assert isinstance(t_dataset.features[0], Tuple)
    assert str(t_dataset.features[0][1]) == str(TIMESERIES(NUMERIC(), NUMERIC()))
    x_imputed = t_dataset.data
    assert len(x_imputed) == 2
    assert len(x_imputed[0]) == 1
    assert len(x_imputed[0][0]) == len(resulting_timesteps)
    assert len(x_imputed[1][0]) == len(resulting_timesteps)
    for instance in x_imputed:
        for idt, timestep in enumerate(resulting_timesteps):
            assert instance[0][idt][0] == timestep


def test_timeseriesimputer_tuple_timeseries():
    check_4_strategies_on(generate_mock_dataset_wmissing(), resulting_timesteps=[0, 1, 2, 3])
    check_4_strategies_on(generate_weird_timestep_timeseries_dataset(), resulting_timesteps=[0, 1, 2, 3, 4, 5, 7])
