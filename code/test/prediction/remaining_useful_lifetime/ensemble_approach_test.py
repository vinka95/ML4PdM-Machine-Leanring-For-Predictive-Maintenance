from typing import List

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from ml4pdm.data import Dataset
from ml4pdm.parsing import DatasetParser
from ml4pdm.prediction import EnsembleApproach, WindowedPredictor
from ml4pdm.transformation import AttributeFilter, SklearnWrapper, WindowingApproach


def test_fit_predict():
    train_dataset, test_dataset = DatasetParser.get_cmapss_data(test=True)
    train_dataset.data = train_dataset.data[:10]
    test_dataset.data = test_dataset.data[:5]
    train_dataset = AttributeFilter.remove_features(train_dataset, [1, 2, 3, 4, 8, 13, 19, 22])
    test_dataset = AttributeFilter.remove_features(test_dataset, [1, 2, 3, 4, 8, 13, 19, 22])

    preprocessing = make_pipeline(SklearnWrapper(MinMaxScaler(), SklearnWrapper.extract_timeseries_concatenated,
                                                 SklearnWrapper.rebuild_timeseries_concatenated), WindowingApproach(1), "passthrough")

    train_instance_lengths = WindowedPredictor.extract_instance_lengths(train_dataset)
    test_instance_lengths = WindowedPredictor.extract_instance_lengths(test_dataset)

    train_dataset = preprocessing.fit_transform(train_dataset)
    test_dataset = preprocessing.transform(test_dataset)

    def annotate_single_timesteps(data: Dataset, instance_lengths: List[int]):
        new_targets = []
        start = 0
        for i, instance_len in enumerate(instance_lengths):
            for j in range(start, start+instance_len):
                if len(data.target) > i:
                    new_targets.append(data.target[i] + instance_len - j)
                else:
                    new_targets.append(instance_len - j)
        data.target = new_targets

    annotate_single_timesteps(train_dataset, train_instance_lengths)
    annotate_single_timesteps(test_dataset, test_instance_lengths)

    rfa = EnsembleApproach(43, DecisionTreeClassifier, fit_preprocessing=EnsembleApproach.random_sampling(14000),
                           max_features=15, splitter="best", criterion="gini")
    rfa.fit(train_dataset)
    predictions = rfa.predict(test_dataset)

    assert len(predictions) == len(test_dataset.data)
    for prediction in predictions:
        assert isinstance(prediction, (float, int))
