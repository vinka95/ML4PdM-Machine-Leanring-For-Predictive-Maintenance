from copy import deepcopy
from random import randrange
from typing import Any, Callable, Dict, List, Type, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.prediction import RemainingUsefulLifetimeEstimator


class EnsembleApproach(RemainingUsefulLifetimeEstimator):
    """An ensemble approach consists of multiple sklearn elements that provide independent RUL predictions.
    These predictions are aggregated in a last step to obtain a single RUL prediction per instance.
    """

    @staticmethod
    def random_sampling(num_samples: int) -> Callable[[Dataset], Dataset]:
        """Returns a function that will randomly sample a dataset until it reaches the given size.
        Duplicates in the resulting dataset are possible.

        :param num_samples: Count of samples to select for the new dataset
        :type num_samples: int
        :return: Randomly sampled Dataset, possibly containing duplicates
        :rtype: Callable[[Dataset], Dataset]
        """
        def function(data: Dataset) -> Dataset:
            new_data = []
            new_target = []
            for _ in range(num_samples):
                idx = randrange(0, len(data.data))
                new_data.append(data.data[idx])
                new_target.append(data.target[idx])
            new_dataset = deepcopy(data)
            new_dataset.data = new_data
            new_dataset.target = new_target
            return new_dataset
        return function

    def __init__(self, n_elements: int, element_class: Type[Union[ClassifierMixin, RegressorMixin]], prediction_aggregation: Callable[[List], Any] = np.mean, fit_preprocessing: Callable[[Dataset], Dataset] = lambda x: x, **element_arguments: Dict) -> None:
        """Initializes an ensemble approach by initializing the child elements with the given arguments.

        :param n_elements: Number of child elements to create
        :type n_elements: int
        :param element_class: Class of the child element that will be instantiated
        :type element_class: Type[Union[ClassifierMixin, RegressorMixin]]
        :param prediction_aggregation: Aggregation function that is used to combine the predictions of every child element
                                       to a single prediction per instance, defaults to np.mean
        :type prediction_aggregation: Callable[[List], Any], optional
        :param fit_preprocessing: Function that is applied to the dataset before a single element is fitted, defaults to identity function
        :type fit_preprocessing: Callable[[Dataset], Dataset], optional
        """
        super().__init__()
        self.element_class = element_class
        self.elements = []
        for _ in range(n_elements):
            self.elements.append(element_class(**element_arguments))
        self.prediction_aggregation = prediction_aggregation
        self.fit_preprocessing = fit_preprocessing

    def fit(self, data: Dataset, label=None, **kwargs) -> "EnsembleApproach":
        """Fits every child element by first applying the preprocessing function and the fitting it on the resulting dataset.

        :param data: Dataset that will be used to fit the child elements after preprocessing
        :type data: Dataset
        :return: Self
        :rtype: EnsembleApproach
        """
        for element in self.elements:
            modified_data = self.fit_preprocessing(data)
            _, feature_data = modified_data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
            feature_data = np.array([instance[0] for instance in feature_data], copy=False)
            element.fit(feature_data, modified_data.target)
        return self

    def predict(self, data: Dataset, **kwargs) -> List[float]:
        """Generates predictions for the dataset by collecting predictions from the child elements and aggregating
        them using the given aggregation function.

        :param data: Dataset that is used to generate the predictions
        :type data: Dataset
        :return: Predictions
        :rtype: List[float]
        """
        _, feature_data = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        feature_data = np.array([instance[0] for instance in feature_data], copy=False)
        predictions = []
        for element in self.elements:
            predictions.append(element.predict(feature_data))
        predictions = list(zip(*predictions))
        return [self.prediction_aggregation(np.array(prediction, copy=False)) for prediction in predictions]
