from ml4pdm.data import Dataset
from ml4pdm.parsing import DatasetParser
from ml4pdm.prediction import Predictor, WindowedPredictor
from ml4pdm.transformation import Transformer, WindowingApproach


class InstanceSizeAsserter(Transformer):
    def __init__(self, lengths: int) -> None:
        super().__init__()
        self.lengths = lengths

    def fit(self, data: Dataset, label=None, **kwargs):
        assert len(data.data) == self.lengths
        return self

    def transform(self, data: Dataset, **kwargs):
        assert len(data.data) == self.lengths
        return data


class ExtractedDataAsserter(InstanceSizeAsserter, Predictor):
    def fit(self, data: Dataset, label=None, **kwargs):
        assert kwargs["extracted_data"].sum()-29*len(kwargs["extracted_data"]) == self.lengths
        return self

    def transform(self, data: Dataset, **kwargs):
        assert kwargs["extracted_data"].sum()-29*len(kwargs["extracted_data"]) == self.lengths
        return data

    def predict(self, data: Dataset, **kwargs):
        return self.transform(data, **kwargs)


def test_transform():
    dataset = DatasetParser.get_cmapss_data()

    expected_length = 0
    for instance in dataset.data:
        expected_length += len(instance[1])-29

    predictor = WindowedPredictor(WindowingApproach(30), [InstanceSizeAsserter(expected_length)], ExtractedDataAsserter(expected_length))
    assert len(predictor.fit_transform(dataset).data) == expected_length
    assert len(predictor.predict(dataset).data) == expected_length
