from typing import Tuple

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.transformation import EMDSignalWrapper


def generate_mock_dataset():
    mock_dataset = Dataset()
    mock_dataset.features = [('ts1', TIMESERIES(NUMERIC(), NUMERIC()))]
    mock_dataset.data = [
        [[(0, 1), (1, 2), (2, 4), (3, 1), (4, 8), (5, -2)]],
        [[(0, 4), (1, 9), (2, 1), (3, 5), (4, 2), (5, 13)]]
    ]
    return mock_dataset


def check_emd_algorithm(wrapper):
    dataset = generate_mock_dataset()
    wrapper.fit(dataset)
    dataset_t = wrapper.transform(dataset)
    assert isinstance(dataset_t, Dataset)
    x_t = dataset_t.data
    assert len(x_t) == 2
    assert len(x_t[0]) == len(wrapper.keep_imfs)
    assert len(x_t[0][0]) == 6
    assert len(x_t[1][0]) == 6
    assert len(dataset_t.features) == len(wrapper.keep_imfs)
    assert isinstance(dataset_t.features[0], Tuple)


def test_emd():
    check_emd_algorithm(EMDSignalWrapper())

    check_emd_algorithm(EMDSignalWrapper(emd_algorithm="emd", keep_components=-1))
    check_emd_algorithm(EMDSignalWrapper(emd_algorithm="eemd", keep_components=[0, -1]))
    check_emd_algorithm(EMDSignalWrapper(emd_algorithm="ceemdan", keep_components=-1))

    check_emd_algorithm(EMDSignalWrapper(emd_algorithm="emd", emd_arguments={"spline_kind": "linear"}, keep_components=-1))
    check_emd_algorithm(EMDSignalWrapper(emd_algorithm="eemd", emd_arguments={"spline_kind": "linear"}, keep_components=-1))
    check_emd_algorithm(EMDSignalWrapper(emd_algorithm="ceemdan", emd_arguments={"spline_kind": "linear"}, keep_components=[12, 1]))
