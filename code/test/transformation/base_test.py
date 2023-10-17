from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.transformation import attach_timesteps, listify_time_series


def generate_mock_dataset():
    mock_dataset = Dataset()
    mock_dataset.features = [('ts1', TIMESERIES(NUMERIC(), NUMERIC()))]
    mock_dataset.data = [
        [[(0, 1), (1, 2), (2, 4)]],
        [[(0, 4), (1, 3), (2, 1)]]
    ]
    return mock_dataset


def generate_mock_dataset_wmissing():
    mock_dataset = Dataset()
    mock_dataset.features = [('ts1', TIMESERIES(NUMERIC(), NUMERIC()))]
    mock_dataset.data = [
        [[(0, 1), (1, 5), (3, 9)]],
        [[(1, 2), (2, -2), (3, 4)]]
    ]
    return mock_dataset


class DatasetMockTuple():
    def __init__(self):
        self.data = [[(1, 1), (2, 300), (3, 3)],
                     [(1, 1), (2, 2), (3, 3), (4, 4), (5, 500)],
                     [(1, 1), (2, 300), (3, 3), (4, 1), (5, 300), (6, 3), (7, 3)]]
        self.target = [1, 2, 3]


class DatasetMockList():
    def __init__(self):
        self.data = [[1, 300, 3],
                     [1, 2, 3, 4, 500],
                     [1, 300, 3, 1, 300, 3, 3]]
        self.target = [1, 2, 3]


def test_listify():
    data_tuple = DatasetMockTuple()
    data_list = DatasetMockList()

    x_tuple_list = listify_time_series(data_tuple.data)

    assert len(x_tuple_list[0]) == 3
    assert len(x_tuple_list[1]) == 5

    x_list_list = listify_time_series(data_list.data)

    for idx, instance in enumerate(x_tuple_list):
        for ididx, value in enumerate(instance):
            assert value == x_list_list[idx][ididx]


def test_attach_timesteps():
    instances = [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]]
    indices = [5, 6, 7, 8, 9, 10]
    res = attach_timesteps(instances, indices)
    assert len(res) == 2
    assert len(res[0]) == 1
    assert len(res[1]) == 1
    assert res[0][0] == [(5, 0), (6, 1), (7, 2), (8, 3), (9, 4), (10, 5)]
