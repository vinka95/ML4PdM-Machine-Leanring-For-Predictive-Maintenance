from copy import deepcopy
from typing import List

import numpy as np

from ml4pdm.data import MULTIDIMENSIONAL, NUMERIC, TIMESERIES, Dataset
from ml4pdm.prediction import HealthIndexEstimator


class HICurveEstimator(HealthIndexEstimator):
    """This HICureEstimator is used to create a health index curve from a list of fixed size feature vectors.
    The estimator stores embeddings of normal operation and computes the minimal distance to obtain a health index.
    """

    def __init__(self, window_size: int) -> None:
        """Initializes the HICurveEstimator.

        :param window_size: The window size that was used to transform the Dataset
        :type window_size: int
        """
        super().__init__()
        self.window_size = window_size
        self.normal_embeddings = []

    def _distance_from_normal(self, embedding: List[float]) -> float:
        """Computes the minimal distance of an embedding from all the embedding that were captured during normal operation.

        :param embedding: Fixed size feature vector that will be used to calculate the distances
        :type embedding: List[float]
        :return: Health index for the machine state from which the embedding was created.
                 0.0 means perfect health and 1.0 means breakdown of the machine.
        :rtype: float
        """
        diffs = [embedding-normal_embedding for normal_embedding in self.normal_embeddings]
        return np.linalg.norm(diffs, axis=1).min()

    def _hi_curve(self, instance: List[List[float]]) -> List[float]:
        """This function maps all the embeddings of a single instance to health indices which creates a HI curve.

        :param instance: A single instance containing multiple embeddings
        :type instance: List[List[float]]
        :return: Health index curve that describes the machine state over time. 0.0 means perfect health and 1.0 means breakdown of the machine.
        :rtype: List[float]
        """
        return list(map(self._distance_from_normal, instance))

    def fit(self, data: Dataset, label=None, **kwargs) -> "HICurveEstimator":
        """When fitting the HICurveEstimator, the first 25% of embeddings for every instance are stored.
        They are assumed to be representing a normal operation and can be used to compute the distances later.

        :param data: Input dataset containing the embeddings for each instance
        :type data: Dataset
        :return: Self
        :rtype: HICurveEstimator
        """
        multi_dim_feature = None
        for i, feature in enumerate(data.features):
            if isinstance(feature[1], MULTIDIMENSIONAL):
                multi_dim_feature = i
                break

        instance_lengths = kwargs["extracted_data"]
        start = 0
        for instance_length in instance_lengths:
            length = instance_length - self.window_size + 1
            for instance in data.data[start:start+int(0.25*length)]:
                self.normal_embeddings.append(instance[multi_dim_feature])
            start += length

        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        """The dataset is transformed by computing the HI curve for every instance.
        This is done by comparing each embedding to the normal embeddings and computing the minimal distance.

        :param data: Dataset to be transformed to HI curves
        :type data: Dataset
        :return: Dataset containing HI curves. 0.0 means perfect health and 1.0 means breakdown of the machine.
        :rtype: Dataset
        """
        multi_dim_feature = None
        for i, feature in enumerate(data.features):
            if isinstance(feature[1], MULTIDIMENSIONAL):
                multi_dim_feature = i
                break

        new_data = deepcopy(data)
        new_data.features[multi_dim_feature] = ("hi_curve", TIMESERIES(NUMERIC(), NUMERIC()))
        new_data.data = []

        instance_lengths = kwargs["extracted_data"]
        start = 0
        transformed_data = []
        for instance_length in instance_lengths:
            length = instance_length - self.window_size + 1
            transformed_data.append([x[multi_dim_feature] for x in data.data[start:start+length]])

            instance = []
            for i, _ in enumerate(data.features):
                if i == multi_dim_feature:
                    instance.append([])
                else:
                    instance.append(data.data[start][i])
            new_data.data.append(instance)
            start += length

        transformed_data = list(map(self._hi_curve, transformed_data))

        for i, transformed in enumerate(transformed_data):
            new_data.data[i][multi_dim_feature] = []
            for j, instance in enumerate(transformed):
                new_data.data[i][multi_dim_feature].append((j, instance))
        return new_data

    def predict(self, data: Dataset, **kwargs) -> Dataset:
        """The dataset is transformed by computing the HI curve for every instance.
        This is done by comparing each embedding to the normal embeddings and computing the minimal distance.

        :param data: Dataset to be transformed to HI curves
        :type data: Dataset
        :return: Dataset containing HI curves. 0.0 means perfect health and 1.0 means breakdown of the machine.
        :rtype: Dataset
        """
        return self.transform(data, **kwargs)
