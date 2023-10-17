import pytest

from ml4pdm.data import ANY, MULTIDIMENSIONAL, NUMERIC, TIMESERIES, AttributeType, Dataset, DatasetSummary
from ml4pdm.parsing import DatasetParser


def test_get_features_of_type():
    non_ts_features = [(0, "id")]
    ts_features = [(1, "setting1"),
                   (2, "setting2"),
                   (3, "setting3"),
                   (4, "s1"),
                   (5, "s2"),
                   (6, "s3"),
                   (7, "s4"),
                   (8, "s5"),
                   (9, "s6"),
                   (10, "s7"),
                   (11, "s8"),
                   (12, "s9"),
                   (13, "s10"),
                   (14, "s11"),
                   (15, "s12"),
                   (16, "s13"),
                   (17, "s14"),
                   (18, "s15"),
                   (19, "s16"),
                   (20, "s17"),
                   (21, "s18"),
                   (22, "s19"),
                   (23, "s20"),
                   (24, "s21")]

    dataset = DatasetParser.get_cmapss_data()

    features = dataset.get_features_of_type(NUMERIC())
    assert features == non_ts_features

    features = dataset.get_features_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
    assert features == ts_features

    features = dataset.get_features_of_type(ANY())
    assert features == non_ts_features + ts_features


def test_get_multivariate_of_type_err():
    dataset = DatasetParser.get_cmapss_data()

    with pytest.raises(LookupError) as excinfo:
        _, _ = dataset.get_multivariate_of_type(MULTIDIMENSIONAL([2, 5]))
    assert "No values found for specified type in this dataset!" in str(excinfo)


def test_get_multivariate_of_type_keep_timestamps():
    dataset = DatasetParser.get_cmapss_data()

    _, data = dataset.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()), True)
    assert len(data) == len(dataset.data)
    assert len(data[0]) == len(dataset.data[0][1])
    assert len(data[0][0]) == 24
    assert len(data[0][0][0]) == 2

    assert data[0][0][0] == dataset.data[0][1][0]
    assert data[0][0][1] == dataset.data[0][2][0]
    assert data[0][0][2] == dataset.data[0][3][0]

    assert data[0][1][0] == dataset.data[0][1][1]
    assert data[0][1][1] == dataset.data[0][2][1]
    assert data[0][1][2] == dataset.data[0][3][1]

    assert data[0][2][0] == dataset.data[0][1][2]
    assert data[0][2][1] == dataset.data[0][2][2]
    assert data[0][2][2] == dataset.data[0][3][2]


def test_get_multivariate_of_type():
    dataset = DatasetParser.get_cmapss_data()

    _, data = dataset.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
    assert len(data) == len(dataset.data)
    assert len(data[0]) == len(dataset.data[0][1])
    assert len(data[0][0]) == 24
    assert isinstance(data[0][0][0], float)

    assert data[0][0][0] == dataset.data[0][1][0][1]
    assert data[0][0][1] == dataset.data[0][2][0][1]
    assert data[0][0][2] == dataset.data[0][3][0][1]

    assert data[0][1][0] == dataset.data[0][1][1][1]
    assert data[0][1][1] == dataset.data[0][2][1][1]
    assert data[0][1][2] == dataset.data[0][3][1][1]

    assert data[0][2][0] == dataset.data[0][1][2][1]
    assert data[0][2][1] == dataset.data[0][2][2][1]
    assert data[0][2][2] == dataset.data[0][3][2][1]


def test_set_from_multivariate_keep_timestamps():
    dataset_new = DatasetParser.get_cmapss_data()
    dataset = DatasetParser.get_cmapss_data()
    features, data = dataset.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()), True)

    for idx, _ in enumerate(dataset.data):
        for feature in features:
            dataset.data[idx][feature[0]] = []

    assert dataset.data != dataset_new.data
    dataset.set_from_multivariate(features, data, False)
    assert dataset.data == dataset_new.data


def test_set_from_multivariate():
    dataset_new = DatasetParser.get_cmapss_data()
    dataset = DatasetParser.get_cmapss_data()
    features, data = dataset.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))

    for idx, _ in enumerate(dataset.data):
        for feature in features:
            dataset.data[idx][feature[0]] = []

    assert dataset.data != dataset_new.data
    dataset.set_from_multivariate(features, data)
    assert dataset.data == dataset_new.data


def test_dataset_getter():
    obj = Dataset()
    obj.data = [[[(0, 1), (1, 1)], 3], [[(0, 2), (1, 8)], 42]]
    obj.features = [('ts', AttributeType.parse('TIMESERIES(NUMERIC:NUMERIC)')), ('best', AttributeType.parse('NUMERIC'))]
    data_list = obj.get_time_series_data_as_array()

    assert len(data_list) == 2
    assert len(data_list[0]) == 2


def test_dataset_simple_cut_generator():
    dataset = DatasetParser.get_cmapss_data()

    prepared_dataset = dataset.generate_simple_cut_dataset(cut_repeats=1, min_length=2, max_length=50)

    p_dat = prepared_dataset.data
    p_tar = prepared_dataset.target

    assert len(p_dat) == 100
    assert min(p_tar) >= 2
    assert max(p_tar) <= 50

    prepared_dataset = dataset.generate_simple_cut_dataset(cut_repeats=2, min_length=11, max_length=133)

    p_dat = prepared_dataset.data
    p_tar = prepared_dataset.target

    assert len(p_dat) == 200
    assert min(p_tar) >= 11
    assert max(p_tar) <= 133


def test_dataset_summary():
    dataset = DatasetParser.get_cmapss_data()

    dataset.data = dataset.data[0:10]

    summary = DatasetSummary()

    result = summary.transform(dataset)

    assert result is not None

    assert result.data is not None

    assert result.features is not None
