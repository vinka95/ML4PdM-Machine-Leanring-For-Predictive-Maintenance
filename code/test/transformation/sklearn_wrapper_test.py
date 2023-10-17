import copy

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from ml4pdm.data import Dataset
from ml4pdm.parsing import DatasetParser
from ml4pdm.transformation import SklearnWrapper


class Passthrough(TransformerMixin):

    def fit(self, x=None):  # pylint: disable=unused-argument
        return self

    def transform(self, data: Dataset):
        return data


def test_fit_transform_1():
    train_dataset = DatasetParser.get_cmapss_data()
    scaling = SklearnWrapper(MinMaxScaler(), SklearnWrapper.extract_timeseries_concatenated, SklearnWrapper.rebuild_timeseries_concatenated)
    train_dataset = scaling.fit_transform(train_dataset)

    for instance in train_dataset.data:
        for sensor in instance[1:]:
            for timestep in sensor:
                assert timestep[1] >= 0.0 and timestep[1] <= 1.0


def test_fit_transform_2():
    train_dataset = DatasetParser.get_cmapss_data()
    old_data = copy.deepcopy(train_dataset.data)
    scaling = SklearnWrapper(Passthrough(), SklearnWrapper.extract_timeseries, SklearnWrapper.rebuild_timeseries)
    train_dataset = scaling.fit_transform(train_dataset)

    assert old_data == train_dataset.data
