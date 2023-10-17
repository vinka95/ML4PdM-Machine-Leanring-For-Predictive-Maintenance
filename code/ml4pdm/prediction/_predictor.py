from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, RegressorMixin

from ml4pdm.data import Dataset


class Predictor(ABC, BaseEstimator, RegressorMixin):
    """Base Predictor class that extends from a sklearn BaseEstimator and a sklearn RegressorMixin.
    """

    @abstractmethod
    def fit(self, data: Dataset, label, **kwargs):
        """This method should train the model of the predictor such that it can predict values afterwards.

        :param data: Training dataset
        :type data: Dataset
        :param label: Labels for training data
        :type label: array-like of float
        """

    @abstractmethod
    def predict(self, data: Dataset, **kwargs):
        """This method should return predictions based on the predictors model for the given Dataset.

        :param data: Dataset that should be used to predict
        :type data: Dataset
        """
