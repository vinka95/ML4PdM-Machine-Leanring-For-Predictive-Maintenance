import copy
from statistics import mean
from typing import List

import numpy as np
import pandas as pd
from tensorflow import keras

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.prediction import RemainingUsefulLifetimeEstimator


class CNNEstimator(RemainingUsefulLifetimeEstimator):
    """This class instantiates a deep CNN architecture with 2 pairs of convolution and pooling layers, followed by a fully connected MLP. 
        Further ideas and formulas for each layer is explained in the below mentioned paper:

    G. Sateesh Babu, Peilin Zhao, and Xiao-Li Li 
    'Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life' 
    Institute for Infocomm Research, A STAR, Singapore
    """

    def __init__(self, kernel_height: int, kernel_width: int, window_size: int, learning_rate=0.1, kernel_size: int = 3, filters: int = 64, units: int = 1, dropout_rate: float = 0.2, activation='relu', epochs: int = 10, batch_size: int = 64, validation_split: float = 0.2, verbose: int = 1) -> None:
        """Initializes the CNN model by instantiating pairs of convolution and pooling layers, followed by a fully connected dense layer and activation.

        :param kernel_height: Height of the kernel space, usually equal to number of features
        :type kernel_height: int
        :param kernel_width: Width of the kernel space
        :type kernel_width: int
        :param window_size: Size of windows the timeseries data going to be split into.
        :type window_size: int
        :param learning_rate: learning rate, defaults to 0.1
        :type learning_rate: float, optional
        :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window, defaults to 3
        :type kernel_size: int, optional
        :param filters: A Tensor. Must have the same type as input, defaults to 64
        :type filters: int, optional
        :param units: 	Dimensionality of the output space, defaults to 1
        :type units: int, optional
        :param dropout_rate: Fraction of the units to drop for the linear transformation of the inputs, defaults to 0.2
        :type dropout_rate: float, optional
        :param activation: Activation function to use, defaults to 'relu'
        :type activation: str, optional
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
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.units = units
        self.filters = filters
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate

        self.model = keras.Sequential([
            keras.layers.Conv2D(self.filters, kernel_size=self.kernel_size, input_shape=(self.kernel_height, self.kernel_width, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=self.activation,
                                use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer='l2', bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),

            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1),
                                   padding='same', data_format=None),

            keras.layers.Conv2D(self.filters, kernel_size=self.kernel_size, padding='valid', data_format=None, dilation_rate=(1, 1), activation=self.activation,
                                use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer='l2', bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),

            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1),
                                   padding='same', data_format=None),

            keras.layers.Flatten(),

            keras.layers.Dropout(self.dropout_rate, noise_shape=None, seed=None),

            keras.layers.Dense(self.units, activation=self.activation, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                               kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None),

            keras.layers.Softmax()
        ])
        self.optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['mae'])

        self.model.summary()

    def fit(self, data: Dataset, label=None,  **kwargs) -> "CNN Estimator Fit":
        """This extends the fit method of CNN model, according to details in the paper mentioned above. 
        Computes the training labels before fitting model and then returns the fitted model.

        :param data: Dataset that should be used for training CNN estimator
        :type data: Dataset
        :return: self
        :rtype: CNNEstimator
        """

        _, train_data = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        data.data = np.array(train_data, copy=False)
        data.target = np.array(data.target, copy=False)
        data.data = np.reshape(data.data, [data.data.shape[0], data.data.shape[2], data.data.shape[1], 1])
        data.target = np.reshape(data.target, [data.target.shape[0], 1])

        return self.model.fit(data.data, data.target, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split,
                              verbose=self.verbose)

    def predict(self, data: Dataset,  **kwargs) -> List[int]:
        """ This extends the predict method of CNN model according to details in the paper mentioned above. 
        Computes the mean of the predictions of each window to get predictions for individual instance and returns them. 

        :param data: Dataset that the RUL predictions should be made for using CNN estimator
        :type data: Dataset
        :return: Predictions
        :rtype: List[int]
        """
        data_copy = copy.deepcopy(data)
        data_df = pd.DataFrame(data_copy.data)
        self.instance_nums = data_df[0]

        _, test_data = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        data.data = np.asarray(test_data)
        data.data = np.reshape(data.data, [data.data.shape[0], data.data.shape[2], data.data.shape[1], 1])

        predictions = self.model.predict(data.data).flatten()

        # Combine Predictions to get exact number of labels
        prediction_tuples = list(zip(self.instance_nums.tolist(), predictions))
        prediction_df = pd.DataFrame(prediction_tuples)

        combined_preds = []
        for i in self.instance_nums.unique():
            preds_ins_df = prediction_df[prediction_df[0] == i]
            instance_preds = []
            num_of_remaining_windows = preds_ins_df.shape[0] - 1
            for j in range(preds_ins_df.shape[0]):
                instance_preds.append(preds_ins_df.iloc[j][1] - (self.window_size * num_of_remaining_windows))
                num_of_remaining_windows = num_of_remaining_windows - 1
            combined_preds.append(mean(instance_preds))

        return combined_preds
