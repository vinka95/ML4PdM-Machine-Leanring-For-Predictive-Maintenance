
from test.transformation.base_test import generate_mock_dataset
from typing import Tuple

from ml4pdm.data import Dataset
from ml4pdm.transformation import PywtWrapper


def test_pywt_wrapper():
    dataset = generate_mock_dataset()

    wrapper = PywtWrapper(interleaving=False)
    data_transformed = wrapper.fit_transform(dataset)
    assert isinstance(data_transformed, Dataset)
    x_transformed = data_transformed.data
    assert len(x_transformed) == 2
    assert len(x_transformed[0]) == 2
    assert len(x_transformed[0][0]) == 2
    assert isinstance(data_transformed.features[0], Tuple)
    assert data_transformed.features[0][0] == 'pywt-approx(ts1)'
    assert data_transformed.features[1][0] == 'pywt-detail(ts1)'

    wrapper = PywtWrapper(interleaving=True)
    data_transformed = wrapper.fit_transform(dataset)
    assert isinstance(data_transformed, Dataset)
    x_transformed = data_transformed.data
    assert len(x_transformed) == 2
    assert len(x_transformed[0]) == 1
    assert len(x_transformed[0][0]) == 4
    assert data_transformed.features[0][0] == 'pywavelet(ts1)'


def test_waveletwrapper_getters():
    assert isinstance(PywtWrapper.modelist(), list)
    assert isinstance(PywtWrapper.wavelist(), list)
