

from operator import itemgetter
from typing import Tuple

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.transformation import FixedSizeFeatureExtractor, TimeSeriesTransformer, UniToMultivariateWrapper


def generate_mixed_dataset():
    dataset = Dataset()
    dataset.data = [[[(1, 1), (2, 300), (3, 3)], 42, [(10, 1), (20, 400), (30, 3), (40, 4)]],
                    [[(1, 1), (2, 2), (3, 3), (4, 4), (5, 500)], 69, [(10, 600), (20, 2), (30, 3), (40, 40), (50, 3), (60, -702)]],
                    [[(1, 1), (2, 300), (3, 3), (4, 1), (5, 300), (6, 3), (7, 3)], 420, [(10, 1), (20, 400), (30, 3), (40, 4), (50, 1), (60, 400), (70, 3), (80, 4)]]]
    dataset.target = [1, 2, 3]
    dataset.features = [('ts1', TIMESERIES(NUMERIC(), NUMERIC())), ('n1', NUMERIC()), ('ts2', TIMESERIES(NUMERIC(), NUMERIC()))]
    return dataset


class TransformerMockSetValue(TimeSeriesTransformer):
    def __init__(self, setvalue=1337):
        super(TransformerMockSetValue).__init__()
        self._setvalue = setvalue

    def fit(self, data: Dataset, label=None, **kwargs) -> "TransformerMockSetValue":
        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        transformed_x = []
        for instance in data.data:
            transformed_instance = []
            for (timestep, _) in instance[0]:
                transformed_instance.append((timestep, self._setvalue))
            transformed_x.append([transformed_instance])
        data.data = transformed_x
        data.features = [('set-to-'+str(self._setvalue), TIMESERIES(NUMERIC(), NUMERIC()))]
        return data


class TransformerMockTimeSeriesLengthAndMax(FixedSizeFeatureExtractor):

    def fit(self, data: Dataset, label=None, **kwargs) -> "TransformerMockTimeSeriesLengthAndMax":
        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        transformed_x = []
        for instance in data.data:
            transformed_instance = [len(instance[0]), max(instance[0], key=itemgetter(1))[1]]
            transformed_x.append(transformed_instance)
        data.data = transformed_x
        data.features = [('len', NUMERIC()), ('max', NUMERIC())]
        return data


def test_unitomultivariatewrapper_timeseriestransformer():
    mixed_dataset = generate_mixed_dataset()
    wrapper1 = UniToMultivariateWrapper(TransformerMockSetValue(setvalue=69), n_jobs=3)
    wrapper1.fit(mixed_dataset)
    transformed_dat = wrapper1.transform(mixed_dataset)
    transformed_x = transformed_dat.data
    transformed_features = transformed_dat.features
    # test feature information
    assert len(transformed_features) == 3
    assert str(transformed_features[0][1]) == 'TIMESERIES(NUMERIC:NUMERIC)'
    assert isinstance(transformed_features[1][1], NUMERIC)
    assert str(transformed_features[2][1]) == 'TIMESERIES(NUMERIC:NUMERIC)'
    assert isinstance(transformed_features[0], Tuple)
    # test number of instances transformed stay the same
    assert len(transformed_x) == 3
    # test number of attributes
    assert len(transformed_x[0]) == 3
    # test that the first and third attribute are still time series of same length
    assert len(transformed_x[0][0]) == 3
    assert len(transformed_x[0][2]) == 4
    assert len(transformed_x[1][0]) == 5
    assert len(transformed_x[1][2]) == 6
    assert len(transformed_x[2][0]) == 7
    assert len(transformed_x[2][2]) == 8
    # test that various values are transformed correctly
    assert transformed_x[0][0][0] == (1, 69)
    assert transformed_x[0][0][1] == (2, 69)
    assert transformed_x[0][2][0] == (10, 69)
    assert transformed_x[0][2][1] == (20, 69)


def test_unitomultivariatewrapper_fixedsizefeatureextractor():
    mixed_dataset = generate_mixed_dataset()
    wrapper2 = UniToMultivariateWrapper(TransformerMockTimeSeriesLengthAndMax())
    wrapper2.fit(mixed_dataset)
    transformed_dataset = wrapper2.transform(mixed_dataset)
    transformed_x = transformed_dataset.data
    transformed_features = transformed_dataset.features
    # test feature information
    assert len(transformed_features) == 5
    for feature_pair in transformed_features:
        assert isinstance(feature_pair[1], NUMERIC)
    # test number of instances transformed stay the same
    assert len(transformed_x) == 3
    # test number of attributes
    assert len(transformed_x[0]) == 5
    # test that various values are extracted correctly
    # extracted values
    assert transformed_x[0][0] == 3
    assert transformed_x[0][3] == 4
    assert transformed_x[1][0] == 5
    assert transformed_x[1][3] == 6
    assert transformed_x[2][0] == 7
    assert transformed_x[2][3] == 8
    # values kept the same
    assert transformed_x[0][2] == 42
    assert transformed_x[1][2] == 69
    assert transformed_x[2][2] == 420
