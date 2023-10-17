import os

import jsonpickle as jp
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from ml4pdm.parsing import DatasetParser
from ml4pdm.transformation import AttributeFilter, RNNAutoencoder, SklearnWrapper, WindowingApproach


def test_fit_transform_predict():
    train_dataset, test_dataset = DatasetParser.get_cmapss_data(test=True)
    train_dataset.data = train_dataset.data[:10]
    test_dataset.data = test_dataset.data[:5]
    train_dataset = AttributeFilter.remove_features(train_dataset, [1, 2, 3, 4, 8, 13, 19, 22])
    test_dataset = AttributeFilter.remove_features(test_dataset, [1, 2, 3, 4, 8, 13, 19, 22])

    preprocessing = make_pipeline(SklearnWrapper(MinMaxScaler(), SklearnWrapper.extract_timeseries_concatenated,
                                                 SklearnWrapper.rebuild_timeseries_concatenated), WindowingApproach(30), "passthrough")

    train_dataset = preprocessing.fit_transform(train_dataset)
    test_dataset = preprocessing.transform(test_dataset)

    rnn_ed = RNNAutoencoder(num_features=16, epochs=2, window_size=30, batch_size=64, verbose=0, plot=True,
                            dropout=0.03263098518086382, learning_rate=0.005840766783749317, units=270)
    rnn_ed.fit(train_dataset)
    plt.close('all')

    filename = "./embed_rul_rnn_ed.json"
    with open(filename, "w") as file:
        file.write(jp.dumps(rnn_ed))
    with open(filename, "r") as file:
        rnn_ed = jp.loads(file.read())
    os.remove(filename)

    transformed = rnn_ed.transform(train_dataset)
    plt.close('all')

    assert len(transformed.data) == len(train_dataset.data)
    for transformed_instance in transformed.data:
        assert len(transformed_instance[1]) == 270

    predicted = rnn_ed.predict(test_dataset)
    plt.close('all')

    assert len(predicted.data) == len(test_dataset.data)
    assert len(predicted.data[1]) == len(test_dataset.data[1])
    assert len(predicted.data[1][1]) == len(test_dataset.data[1][1])
    assert len(predicted.data[1][1][1]) == len(test_dataset.data[1][1][1])
