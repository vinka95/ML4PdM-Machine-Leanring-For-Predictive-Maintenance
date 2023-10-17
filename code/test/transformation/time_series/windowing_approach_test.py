from ml4pdm.parsing import DatasetParser
from ml4pdm.transformation import WindowingApproach


def test_windowing():
    assert WindowingApproach(2)._windowing([1, 2, 3, 4]) == [[1, 2], [2, 3], [3, 4]]
    assert WindowingApproach(2, 2)._windowing([1, 2, 3, 4]) == [[1, 2], [3, 4]]
    assert WindowingApproach(4)._windowing([1, 2, 3, 4]) == [[1, 2, 3, 4]]


def test_transform():
    dataset_new = DatasetParser.get_cmapss_data()
    dataset = DatasetParser.get_cmapss_data()

    expected_length = 0
    for instance in dataset.data:
        expected_length += len(instance[1])-29

    dataset = WindowingApproach(30).fit_transform(dataset)

    assert len(dataset.data) == expected_length

    assert dataset.data[0][0] == dataset_new.data[0][0]
    assert dataset.data[50][0] == dataset_new.data[0][0]
    assert dataset.data[200][0] == dataset_new.data[1][0]
    assert dataset.data[-1][0] == dataset_new.data[-1][0]

    for instance in dataset.data:
        assert len(instance) == 25
        for fidx in range(1, 25):
            assert len(instance[fidx]) == 30
