from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin

from ml4pdm.data import Dataset


class Transformer(ABC, BaseEstimator, TransformerMixin):
    """Base Transformer class that extends from sklearn BaseEstimator and TransformerMixin.

    :param BaseEstimator: Base class for all estimators in scikit-learn.
    :type BaseEstimator: sklearn.base.BaseEstimator
    :param TransformerMixin: Mixin class for all transformers in scikit-learn.
    :type TransformerMixin: sklearn.base.TransformerMixin
    """
    @abstractmethod
    def fit(self, data: Dataset, label, **kwargs):
        """This method should train the model of the transformer such that it can transform the values afterwards.
        :param data: Input samples.
        :type data: array-like of shape (n_samples, n_features)
        :param label: Target values.
        :type label: array-like of shape (n_samples,)
        """
    @abstractmethod
    def transform(self, data: Dataset, **kwargs):
        """This method should return transformation based on the transformer model for the given Dataset.
        :param data: Input samples.
        :type data: array-like of shape (n_samples, n_features)
        """
