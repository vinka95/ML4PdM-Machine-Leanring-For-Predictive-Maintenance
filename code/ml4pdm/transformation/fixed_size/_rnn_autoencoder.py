from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from tensorflow import keras

from ml4pdm.data import ANY, MULTIDIMENSIONAL, NUMERIC, TIMESERIES, Dataset
from ml4pdm.transformation import FixedSizeFeatureExtractor


class RNNAutoencoder(FixedSizeFeatureExtractor):
    """This class models a RNN Autoencoder. The model consists of an encoder that maps a window of timeseries data on a fixed size
    feature vector called embedding. The model also contains a decoder that can transform the embedding to a timeseries. This way it can be
    trained to obtain highly representative embeddings for a given timeseries.
    """

    def __init__(self, num_features: int, units: int, learning_rate: float, window_size: int,
                 dropout: float, epochs: int, batch_size: int, verbose=0, plot=False) -> None:
        """Initializes the RNN AE by creating the underlying keras model which consists of five layers. These resemble an encoder and
        decoder model.

        :param num_features: Count of feature values for a single timestem (e.g. count of sensors)
        :type num_features: int
        :param units: Size of embedding vector and number of GRU units
        :type units: int
        :param learning_rate: Learning rate used by the Adam optimizer
        :type learning_rate: float
        :param window_size: Window size that was used to transform the dataset
        :type window_size: int
        :param dropout: Dropout used in the GRU
        :type dropout: float
        :param epochs: Number of epochs that the fit will train the models for
        :type epochs: int
        :param batch_size: Batch size used when training the model
        :type batch_size: int
        :param verbose: Verbosity of the keras training, defaults to 0
        :type verbose: int, optional
        :param plot: Defines whether there should be additional plots that show the training progress and the reconstructed timeseries in
                     comparison to the original one, defaults to False
        :type plot: bool, optional
        """
        super().__init__()

        layer1 = keras.layers.InputLayer(input_shape=(window_size, num_features))
        layer2 = keras.layers.GRU(units, dropout=dropout)
        layer3 = keras.layers.RepeatVector(window_size)
        layer4 = keras.layers.GRU(units, dropout=dropout, return_sequences=True)
        layer5 = keras.layers.TimeDistributed(keras.layers.Dense(num_features))

        self.encoder = keras.Sequential([layer1, layer2], "Encoder")
        self.model = keras.Sequential([layer1, layer2, layer3, layer4, layer5], "RNN-ED")

        adam = keras.optimizers.Adam(learning_rate)
        self.model.compile(loss="mse", optimizer=adam, metrics=["mae"])

        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.plot = plot
        self.num_features = num_features
        self.units = units
        self.window_size = window_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.fitted = False

    def fit(self, data: Dataset, label=None, **kwargs) -> "RNNAutoencoder":
        """Trains the RNN AE model by trying to match the reconstructed timeseries to the original as best as possible.
        This way it can be ensured that the internal representation / embedding represents the timeseries very well.
        The training history will be plotted if the 'plot' member variable was set to true.

        :param data: Input dataset containing the original timeseries to be reconstructed
        :type data: Dataset
        :return: Self
        :rtype: RNNAutoencoder
        """
        if self.fitted:
            return self

        callbacks = None
        if "callbacks" in kwargs:
            callbacks = kwargs["callbacks"]

        _, ts_data = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        ts_data = np.array(ts_data, copy=False)
        train_history = self.model.fit(ts_data, ts_data,
                                       epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                                       validation_split=0.2, callbacks=callbacks).history
        self.fitted = True

        if self.plot:
            plt.figure(figsize=(14, 9))
            plt.plot(train_history["loss"], label="train")
            plt.plot(train_history["val_loss"], label="val")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend()
            plt.show(block=False)
        return self

    def predict(self, data: Dataset, **kwargs) -> Dataset:  # pylint: disable=unused-argument
        """This function will use the input timeseries to generate embeddings and then reconstruct the timeseries again.
        The reconstruction and original timeseries will be plotted if the 'plot' member variable was set to true.

        :param data: Input dataset containing the original timeseries to be reconstructed
        :type data: Dataset
        :return: Reconstructed timeseries
        :rtype: Dataset
        """
        features, ts_data = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        ts_data = np.array(ts_data, copy=False)
        reconstructed = self.model.predict(ts_data)

        if self.plot:
            for idx in range(15):
                plt.figure(figsize=(14, 9))
                for i in range(0, 100, 50):
                    tr_dat = [x[idx] for x in ts_data[i]]
                    pr_dat = [x[idx] for x in reconstructed[i]]
                    plt.plot(tr_dat, label="instance {}: true".format(i))
                    plt.plot(pr_dat, label="instance {}: pred, mae: {}".format(i, mean_absolute_error(tr_dat, pr_dat)))
                plt.legend()
                plt.title("ID: {}".format(idx))
                plt.show(block=False)

        data.set_from_multivariate(features, reconstructed)
        return data

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        """This function will transform the timeseries into a smaller fixed size feature vector / embedding.
        The reconstruction and original timeseries will be plotted if the 'plot' member variable was set to true.

        :param data: Input timeseries that will be transformed to embeddings
        :type data: Dataset
        :return: Dataset containing the generated embeddings
        :rtype: Dataset
        """
        if self.plot:
            reconstructed = self.predict(data)
            _, ts_reconstructed = reconstructed.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
            _, ts_data = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
            for idx in range(15):
                plt.figure(figsize=(14, 9))
                for i in range(0, 100, 20):
                    tr_dat = [y[idx] for y in ts_data[i]]
                    pr_dat = [y[idx] for y in ts_reconstructed[i]]
                    plt.plot(tr_dat, label="instance {}: true".format(i))
                    plt.plot(pr_dat, label="instance {}: pred, mae: {}".format(i, mean_absolute_error(tr_dat, pr_dat)))
                plt.legend()
                plt.title("ID: {}".format(idx))
                plt.show(block=False)

        features, ts_data = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        ts_data = np.array(ts_data, copy=False)
        embeddings = self.encoder.predict(ts_data)

        new_features = []
        for feature in data.get_features_of_type(ANY()):
            if not feature in features:
                new_features.append(feature)

        new_data = []
        for idx, embedding in enumerate(embeddings):
            new_data.append([])
            for feature in new_features:
                new_data[idx].append(data.data[idx][feature[0]])
            new_data[idx].append(embedding)

        new_features.append(("embeddings", MULTIDIMENSIONAL([self.units])))

        data.features = new_features
        data.data = new_data
        return data

    def __getstate__(self) -> Dict:
        """Saves the current state into a dictionary and returns it. This is useful for serialization.

        :return: Dictionary containing the current state
        :rtype: Dict
        """
        state = self.__dict__.copy()
        state['model'] = {"config": self.model.get_config(), "weights": self.model.get_weights()}
        state['encoder'] = {"config": self.encoder.get_config(), "weights": self.encoder.get_weights()}
        return state

    def __setstate__(self, state) -> None:
        """Restores the state from a dictionary. This is useful for de-serialization.

        :param state: State that will be parsed
        :type state: Dict
        """
        self.__dict__ = state.copy()
        model_weights = state["model"]["weights"]
        self.model = keras.Sequential.from_config(state["model"]["config"])
        self.model.set_weights(model_weights)

        encoder_weights = state["encoder"]["weights"]
        self.encoder = keras.Sequential.from_config(state["encoder"]["config"])
        self.encoder.set_weights(encoder_weights)
