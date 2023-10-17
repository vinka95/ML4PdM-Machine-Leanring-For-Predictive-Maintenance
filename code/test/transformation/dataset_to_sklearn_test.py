
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from ml4pdm.data import NUMERIC, Dataset
from ml4pdm.transformation import (DatasetToSklearn, EMDSignalWrapper, ML4PdM, PytsSupportedAlgorithm, PytsTransformWrapper, PywtWrapper,
                                   TimeSeriesImputer, UniToMultivariateWrapper)

from .base_test import generate_mock_dataset
from .fixed_size._pyts_transform_test import generate_synthetic_dataset


def generate_simple_dataset():
    dataset = Dataset()
    dataset.data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    dataset.features = [('n1', NUMERIC()), ('n2', NUMERIC()), ('n3', NUMERIC()), ('n4', NUMERIC())]
    return dataset


def extractor_mock_dataset_convenient_function(data: Dataset):
    return data.get_time_series_data_as_array()


def test_dataset_to_sklearn_and_ml4pdm_wrapper():

    # test default extractor
    dataset_numeric = generate_simple_dataset()
    print('original', dataset_numeric.data)
    dts = DatasetToSklearn()

    res_numeric = dts.fit_transform(dataset_numeric)
    assert len(res_numeric) == 3
    assert len(res_numeric[0]) == 4

    dataset_ts = generate_mock_dataset()

    # with custom extractor and ml4pdm wrapper
    ml4pdm = ML4PdM(ml4pdm_steps=[EMDSignalWrapper(keep_components=[0, 1])], extractor=extractor_mock_dataset_convenient_function)
    res_ts = ml4pdm.fit_transform(dataset_ts)
    assert len(res_ts) == 2
    assert len(res_ts[0]) == 2
    assert len(res_ts[0][0]) == 3
    print('ml4pdm-test', res_ts)


def test_bigger_pipeline():
    dataset = generate_synthetic_dataset(n_instances=5)
    print('dataset done')
    pipeline = make_pipeline(TimeSeriesImputer(), EMDSignalWrapper(keep_components=[0, 1]), UniToMultivariateWrapper(PytsTransformWrapper(
        PytsSupportedAlgorithm.BOP)), DatasetToSklearn(), RandomForestRegressor(n_estimators=3, max_depth=2))
    print('pipeline done')
    pipeline.fit(dataset, dataset.target)
    print('fitting done')
    y_pred = pipeline.predict(dataset)
    assert len(y_pred) == len(dataset.target)
    print('y_pred', y_pred)
    print('y_true', dataset.target)

    pipeline2 = make_pipeline(ML4PdM(ml4pdm_steps=[TimeSeriesImputer(), PywtWrapper(), UniToMultivariateWrapper(PytsTransformWrapper(
        PytsSupportedAlgorithm.WEASEL))]), RandomForestRegressor(n_estimators=5, max_depth=1))

    pipeline2.fit(dataset, dataset.target)

    y_pred2 = pipeline2.predict(dataset)
    print('2nd-pipeline', y_pred2)

    assert len(y_pred2) == len(dataset.target)
