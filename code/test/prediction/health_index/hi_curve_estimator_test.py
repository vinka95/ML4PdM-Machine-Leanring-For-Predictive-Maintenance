from numpy import float32
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from ml4pdm.parsing import DatasetParser
from ml4pdm.prediction import HICurveEstimator, WindowedPredictor
from ml4pdm.transformation import AttributeFilter, RNNAutoencoder, SklearnWrapper, WindowingApproach


def test_fit_predict():
    train_dataset, test_dataset = DatasetParser.get_cmapss_data(test=True)
    train_dataset.data = train_dataset.data[:10]
    test_dataset.data = test_dataset.data[:5]
    train_dataset = AttributeFilter.remove_features(train_dataset, [1, 2, 3, 4, 8, 13, 19, 22])
    test_dataset = AttributeFilter.remove_features(test_dataset, [1, 2, 3, 4, 8, 13, 19, 22])

    pipeline = make_pipeline(SklearnWrapper(MinMaxScaler(), SklearnWrapper.extract_timeseries_concatenated,
                                            SklearnWrapper.rebuild_timeseries_concatenated),
                             WindowedPredictor(WindowingApproach(30),
                                               [RNNAutoencoder(num_features=16, epochs=2, window_size=30, batch_size=64, verbose=0, plot=False,
                                                               dropout=0.03263098518086382, learning_rate=0.005840766783749317, units=270)],
                                               HICurveEstimator(30)))

    pipeline.fit(train_dataset)
    predicted = pipeline.predict(test_dataset)

    assert len(predicted.data) == len(test_dataset.data)
    assert len(predicted.data[1][1]) == len(test_dataset.data[1][1])-29
    for health_index in predicted.data[1][1]:
        assert isinstance(health_index[1], (float, float32))
        assert health_index[1] >= 0.0
        assert health_index[1] <= 1.0
