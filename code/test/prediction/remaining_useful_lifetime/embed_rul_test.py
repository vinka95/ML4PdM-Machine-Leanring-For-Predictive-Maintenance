from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from ml4pdm.parsing import DatasetParser
from ml4pdm.prediction import EmbedRUL
from ml4pdm.transformation import AttributeFilter, RNNAutoencoder, SklearnWrapper, WindowingApproach


def test_fit_predict():
    train_dataset, test_dataset = DatasetParser.get_cmapss_data(test=True)
    train_dataset.data = train_dataset.data[:10]
    test_dataset.data = test_dataset.data[:5]
    train_dataset = AttributeFilter.remove_features(train_dataset, [1, 2, 3, 4, 8, 13, 19, 22])
    test_dataset = AttributeFilter.remove_features(test_dataset, [1, 2, 3, 4, 8, 13, 19, 22])

    preprocessing = make_pipeline(SklearnWrapper(MinMaxScaler(), SklearnWrapper.extract_timeseries_concatenated,
                                                 SklearnWrapper.rebuild_timeseries_concatenated), "passthrough")

    train_dataset = preprocessing.fit_transform(train_dataset)
    test_dataset = preprocessing.transform(test_dataset)

    embed_rul = EmbedRUL.from_params(num_features=16, epochs=2)
    embed_rul.fit(train_dataset)
    predicted = embed_rul.predict(test_dataset)

    assert len(predicted) == len(test_dataset.data)
    for prediction in predicted:
        assert isinstance(prediction, (float, int))
        assert prediction >= 0
        assert prediction <= 120


def test_fit_predict_pretrained():
    train_dataset, test_dataset = DatasetParser.get_cmapss_data(test=True)
    train_dataset.data = train_dataset.data[:10]
    test_dataset.data = test_dataset.data[:5]
    train_dataset = AttributeFilter.remove_features(train_dataset, [1, 2, 3, 4, 8, 13, 19, 22])
    test_dataset = AttributeFilter.remove_features(test_dataset, [1, 2, 3, 4, 8, 13, 19, 22])

    preprocessing = make_pipeline(SklearnWrapper(MinMaxScaler(), SklearnWrapper.extract_timeseries_concatenated,
                                                 SklearnWrapper.rebuild_timeseries_concatenated), "passthrough")

    train_dataset = preprocessing.fit_transform(train_dataset)
    test_dataset = preprocessing.transform(test_dataset)

    rnn_ed = RNNAutoencoder(num_features=16, epochs=2, window_size=30, batch_size=64, verbose=0, plot=False,
                            dropout=0.03263098518086382, learning_rate=0.005840766783749317, units=270)

    train_dataset2 = WindowingApproach(30).transform(train_dataset)
    rnn_ed.fit(train_dataset2)

    embed_rul = EmbedRUL.from_params_pretrained_rnn(rnn_ed, max_time_lag=150)
    embed_rul.fit(train_dataset)
    predicted = embed_rul.predict(test_dataset)

    assert len(predicted) == len(test_dataset.data)
    for prediction in predicted:
        assert isinstance(prediction, (float, int))
        assert prediction >= 0
        assert prediction <= 120
