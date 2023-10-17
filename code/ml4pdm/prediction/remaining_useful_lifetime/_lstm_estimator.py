import copy
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tensorflow import keras

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.prediction import RemainingUsefulLifetimeEstimator


class LSTMEstimator(RemainingUsefulLifetimeEstimator):
    """This class instantiates a LSTM network. This LSTM model is made up of multiple dense and dropout layers
        following the architecture explained in the below mentioned paper:

        S. Zheng, K. Ristovski, A. Farahat and C. Gupta, 
        "Long Short-Term Memory Network for Remaining Useful Life estimation," 
        2017 IEEE International Conference on Prognostics and Health Management (ICPHM), 
        2017, pp. 88-95, doi: 10.1109/ICPHM.2017.7998311.
    """

    def __init__(self, window_size: int, num_features: int, units: int = 100, return_sequences: bool = True, dropout_rate: float = 0.2, activation='linear', learning_rate=0.1, epochs: int = 10, batch_size: int = 64, validation_split: float = 0.2, verbose: int = 1) -> None:
        """ Initializes the LSTM by creating the underlying keras model which consists of six layers.

        :param window_size: Size of windows the timeseries data going to be split into.
        :type window_size: int
        :param num_features: Number of features in the given data, determines the shape of the input layer.
        :type num_features: int
        :param units: Positive integer, dimensionality of the output space, defaults to 100
        :type units: int, optional
        :param return_sequences: Determines whether to return the last output, defaults to True
        :type return_sequences: bool, optional
        :param dropout_rate: Fraction of the units to drop for the linear transformation of the inputs, defaults to 0.2
        :type dropout_rate: float, optional
        :param activation: Activation function to use, defaults to 'linear'
        :type activation: str, optional
        :param learning_rate: [description], defaults to 0.1
        :type learning_rate: float, optional
        :param epochs: Determines number of samples per gradient update., defaults to 10
        :type epochs: int, optional
        :param batch_size: Number of samples per batch of computation., defaults to 64
        :type batch_size: int, optional
        :param validation_split: Fraction of the training data to be used as validation data., defaults to 0.2
        :type validation_split: float, optional
        :param verbose: Verbosity mode, defaults to 1
        :type verbose: int, optional
        """
        super().__init__()

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.window_size = window_size
        self.num_features = num_features
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.units = units
        self.return_sequences = return_sequences
        self.learning_rate = learning_rate

        self.model = keras.Sequential([
            keras.layers.LSTM(input_shape=(self.window_size, self.num_features), units=self.units,
                              return_sequences=self.return_sequences, name="lstm_0"),
            keras.layers.Dropout(rate=self.dropout_rate, name="dropout_0"),
            keras.layers.LSTM(units=50, return_sequences=True, name="lstm_1"),
            keras.layers.Dropout(rate=self.dropout_rate, name="dropout_1"),
            keras.layers.LSTM(units=25, return_sequences=False, name="lstm_2"),
            keras.layers.Dropout(rate=self.dropout_rate, name="dropout_2"),
            keras.layers.Dense(units=1, name="dense_0"),
            keras.layers.Activation(self.activation, name="activation_0")
        ])

        self.optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['mae'])

    def fit(self, data: Dataset, label=None,  **kwargs) -> "LSTM Estimator Fit":
        """This extends the fit method of LSTM model, according to details in the paper mentioned above. 
        Computes the training labels before fitting model and then returns the fitted model.

        :param data: Dataset that should be used for training LSTM estimator
        :type data: Dataset
        :return: self
        :rtype: LSTMEstimator
        """

        _, train_data = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        data.data = np.array(train_data, copy=False)
        data.target = np.array(data.target, copy=False)

        return self.model.fit(data.data, data.target, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split,
                              verbose=self.verbose)

    def predict(self, data: Dataset,  **kwargs) -> List[int]:
        """ This extends the predict method of LSTM model according to details in the paper mentioned above. 
        Computes the mean of the predictions of each window to get predictions for individual instance and returns them. 

        :param data: Dataset that the RUL predictions should be made for using LSTM estimator
        :type data: Dataset
        :return: Predictions
        :rtype: List[int]
        """
        data_copy = copy.deepcopy(data)
        data_df = pd.DataFrame(data_copy.data)
        self.instance_nums = data_df[0]

        _, test_data = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        data.data = np.asarray(test_data)
        data.target = np.asarray(data.target)

        predictions = self.model.predict(data.data).flatten()

        # Combine Predictions to get exact number of labels
        prediction_tuples = list(zip(self.instance_nums.tolist(), predictions))
        prediction_df = pd.DataFrame(prediction_tuples)

        combined_predictions = []
        for i in self.instance_nums.unique():
            pred_instance_df = prediction_df[prediction_df[0] == i]
            instance_preds = []
            num_of_remaining_windows = pred_instance_df.shape[0] - 1
            for j in range(pred_instance_df.shape[0]):
                instance_preds.append(pred_instance_df.iloc[j][1] - (self.window_size * num_of_remaining_windows))
                num_of_remaining_windows = num_of_remaining_windows - 1
            combined_predictions.append(mean(instance_preds))

        return combined_predictions
