import math
import random
from typing import Tuple

import numpy as np
import pytest

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.transformation import PytsSupportedAlgorithm, PytsTransformWrapper, TimeSeriesImputer, attach_timesteps


def generate_big_dataset():
    dataset = Dataset()
    dataset.data = [[[(1, 1), (2, 300), (3, 3), (4, 1), (5, 300), (6, 3), (7, 3)]],
                    [[(1, 2), (2, 300), (3, 45), (4, 1), (5, 300), (6, 2), (7, 17)]],
                    [[(1, 3), (2, 300), (3, 3), (4, 9), (5, 300), (6, 6), (7, 3)]],
                    [[(1, 4), (2, -50), (3, 2), (4, 1), (5, 300), (6, 4), (7, 8)]],
                    [[(1, 5), (2, 300), (3, 3), (4, 1), (5, 300), (6, 6), (7, 3)]]
                    ]
    dataset.target = [1, 2, 3, 4, 1]
    dataset.features = [('ts1', TIMESERIES(NUMERIC(), NUMERIC()))]
    return dataset


def generate_synthetic_dataset(n_instances=100, min_length_instance=50, random_state=69):
    random.seed(random_state)
    X = []
    for _ in range(1000):
        ts = []
        runtime = random.randrange(150)+200
        for i in range(runtime):
            j = i-random.random()
            ts.append(math.sin(j**2)*(runtime-i)/runtime)
        X.append(ts)
    sampled_x = []
    sampled_y = []
    for i in range(n_instances):
        instance = np.asarray(random.choice(X), dtype=np.float)
        instance_length = len(instance)
        proportion = random.random()
        n_step_cut = math.floor(proportion*instance_length)
        if instance_length - n_step_cut < min_length_instance:
            n_step_cut = instance_length - min_length_instance
        cut_instance = instance[:instance_length-n_step_cut]
        sampled_x.append([cut_instance])
        sampled_y.append(n_step_cut)
    dataset = Dataset()
    dataset.data = attach_timesteps(np.asarray(sampled_x, dtype=object))
    dataset.target = np.asarray(sampled_y)
    dataset.features = [('ts1', TIMESERIES(NUMERIC(), NUMERIC()))]
    dataset.target_name = 'ttf'
    dataset.target_type = NUMERIC()

    return dataset


def test_pytstransform_different_algorithms():
    dataset = generate_synthetic_dataset(n_instances=5)
    imputed_dataset = (TimeSeriesImputer()).fit_transform(dataset)

    # BOP
    pyts_wrapper = PytsTransformWrapper(PytsSupportedAlgorithm.BOP, algorithm_params={"window_size": 9, "word_size": 3, "n_bins": 2})
    transformed_dataset = pyts_wrapper.fit_transform(imputed_dataset)
    assert isinstance(transformed_dataset, Dataset)
    transformed_x = transformed_dataset.data
    assert transformed_x.shape == (5, 6)
    assert len(transformed_dataset.features) == 6
    assert isinstance(transformed_dataset.features[0], Tuple)

    # BOSS
    pyts_wrapper = PytsTransformWrapper(PytsSupportedAlgorithm.BOSS, algorithm_params={"window_size": 9, "word_size": 3, "n_bins": 2})
    transformed_dataset = pyts_wrapper.fit_transform(imputed_dataset)
    assert isinstance(transformed_dataset, Dataset)
    transformed_x = transformed_dataset.data
    assert transformed_x.shape == (5, 8)
    assert len(transformed_dataset.features) == 8

    # ROCKET
    pyts_wrapper = PytsTransformWrapper(PytsSupportedAlgorithm.ROCKET, algorithm_params={"n_kernels": 20, "random_state": 69})
    transformed_dataset = pyts_wrapper.fit_transform(imputed_dataset)
    assert isinstance(transformed_dataset, Dataset)
    transformed_x = transformed_dataset.data
    assert transformed_x.shape == (5, 40)
    assert len(transformed_dataset.features) == 40

    # SHAPELET
    pyts_wrapper = PytsTransformWrapper(PytsSupportedAlgorithm.SHAPELET, algorithm_params={
        "window_sizes": [12, 42], "criterion": "anova", "sort": True, "random_state": 420})
    transformed_dataset = pyts_wrapper.fit_transform(imputed_dataset)
    assert isinstance(transformed_dataset, Dataset)
    transformed_x = transformed_dataset.data
    assert transformed_x.shape == (5, 40)
    assert len(transformed_dataset.features) == 40

    # WEASEL
    pyts_wrapper = PytsTransformWrapper(PytsSupportedAlgorithm.WEASEL, algorithm_params={"word_size": 2, "n_bins": 2, "window_sizes": [3, 5, 7]})
    transformed_dataset = pyts_wrapper.fit_transform(imputed_dataset)
    assert isinstance(transformed_dataset, Dataset)
    transformed_x = transformed_dataset.data
    assert transformed_x.shape == (5, 35)
    assert len(transformed_dataset.features) == 35


test_pytstransform_different_algorithms()


def test_differeing_lengths_timeseries():
    dataset = generate_synthetic_dataset(n_instances=20)
    wrapper = PytsTransformWrapper(PytsSupportedAlgorithm.BOP)
    with pytest.raises(TypeError):
        wrapper.fit(dataset)


def test_constructor_wrong_attribute():
    with pytest.raises(AttributeError):
        PytsTransformWrapper(algorithm=PytsSupportedAlgorithm)
