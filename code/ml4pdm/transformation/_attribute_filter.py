from copy import deepcopy
from typing import List

import numpy as np

from ml4pdm.data import Dataset
from ml4pdm.transformation import Transformer


class AttributeFilter(Transformer):
    """This class is used as a pipeline element to remove certain attributes completely from input dataset.
    This can either be directly specified by a list of indices to be removed
    or by a condition on the number of unique values found in the attribute over the whole dataset.
    Attributes which have only one unique value of the whole dataset don't contribute any information to the learning task,
    hence why they can be omitted without loss of information.
    The unique values are counted for every attribute type.
    Specifically for TIMESERIES attributes the unique values are counted over the whole time series,
    such that a constant as a time series counts as one unique value for the instance.
    """

    def __init__(self, remove_indices: List[int] = None, min_unique_values: int = 2):
        """Constructor which sets the directly removed attribute indices and the minimum required unique values.

        :param remove_indices: directly removed indices, defaults to None
        :type remove_indices: List[int], optional
        :param min_unique_values: minimum number of unique values required to not be removed by this filter, defaults to 2
        :type min_unique_values: int, optional
        """
        self.remove_indices = remove_indices or []
        self.min_unique_values = min_unique_values
        self.all_removed_indices = []

    @staticmethod
    def remove_features(dataset: Dataset, remove_indices: List[int] = None) -> Dataset:
        """Removes the attributes specified by the remove_indices parameter from a copy of the specified dataset.
        The original dataset is not changed.

        :param dataset: dataset on which the attributes are to be removed
        :type dataset: Dataset
        :param remove_indices: list of indices for the attributes to be removed, defaults to None
        :type remove_indices: List[int], optional
        :return: dataset containing only the not removed attributes
        :rtype: Dataset
        """
        remove_indices = remove_indices or []
        res_dataset = deepcopy(dataset)
        data = res_dataset.data
        features = res_dataset.features
        for index in sorted(remove_indices, reverse=True):
            for instance in data:
                del instance[index]
            del features[index]
        return res_dataset

    def fit(self, data: Dataset, label=None, **kwargs) -> "AttributeFilter":
        """Fits this filter object on the specified dataset.
        The indices of the attributes which don't satisfy the condition are calculated and stored.

        :param data: dataset to fit this filter on
        :type data: Dataset
        :return: Self
        :rtype: AttributeFilter
        """
        data_list = data.get_time_series_data_as_array()
        condition_indices = [i for i in range(len(data_list[0])) if len(np.unique(sum([list(np.unique(instance[i]))
                                                                                  for instance in data_list], []))) < self.min_unique_values]
        self.all_removed_indices = self.remove_indices + condition_indices
        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        """Removes the predefined attributes from a copy of the specified dataset and returns it.

        :param data: dataset to remove the predefined attributes from
        :type data: Dataset
        :return: dataset with the predefined attributes removed
        :rtype: Dataset
        """
        return AttributeFilter.remove_features(data, self.all_removed_indices)
