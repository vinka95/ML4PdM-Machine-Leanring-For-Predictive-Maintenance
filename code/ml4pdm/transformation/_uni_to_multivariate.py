'''This module contains the UniToMultivariateWrapper which is used to apply a transformation for univariate timeseries to multivariate timeseries.
'''
from copy import copy, deepcopy
from itertools import starmap
from multiprocessing import Pool

from ml4pdm.data import TIMESERIES, Dataset
from ml4pdm.transformation import Transformer


def _single_fit(dataset: Dataset, is_timeseries: bool, transformer_object: Transformer, label) -> "Transformer":
    if not is_timeseries:
        return None
    transformer = deepcopy(transformer_object)
    return transformer.fit(dataset, label)


def _single_transform(dataset: Dataset, is_timeseries: bool, transformer_object: Transformer) -> "Dataset":
    if not is_timeseries:
        return dataset
    return transformer_object.transform(dataset)


class UniToMultivariateWrapper(Transformer):
    """This class is used to apply a transformer of univariate time series to all timeseries features of all instances in a dataset.
    This will create a seperate transformer object for each TIMESERIES attribute in the Dataset.
    These transformer objects are then fitted accordingly on those TIMESERIES seperately.
    At the end we put the instances with the transformed TIMESERIES attributes back together.
    The non-TIMESERIES attributes will be untouched and are retained in the resulting dataset.
    """

    def __init__(self, transformer, n_jobs=1):
        """Constructor for the UniToMultivariateWrapper with attributes for the transformer.

        :param transformer: the transformer to be used on every TIMESERIES feature
        :type transformer: obj containing fit and transform method
        """
        super(UniToMultivariateWrapper).__init__()
        self.transformer = transformer
        self._attr_is_timeseries = None
        self._transformer_list = []
        self.n_jobs = n_jobs

    def _split_attributes(self, data: Dataset):
        """splits the given dataset along its features and returns a list of dataset,
        each containing just one feature.

        :param data: Dataset to be split
        :type data: Dataset
        :return: List of datasets containing single features
        :rtype: List[Dataset]
        """
        attr_x = [[] for _ in range(len(data.features))]

        for instance in data.data:
            for idx, feature in enumerate(instance):
                attr_x[idx].append([feature])

        attr_datasets = []
        for i, attr in enumerate(attr_x):
            dataset = copy(data)
            dataset.data = attr
            dataset.features = [data.features[i]]
            attr_datasets.append(dataset)

        return attr_datasets

    def fit(self, data: Dataset, label=None, **kwargs):
        """This method creates a transformer object for each TIMESERIES attribute of the instances and fits each transformer on its subset of the features.

        :param data: Dataset for fitting this UniToMultivariateWrapper
        :type data: Dataset
        :param label: ignored, defaults to None
        :type label: arbitrary, optional
        :return: Self
        :rtype: UniToMultivariateWrapper
        """
        self._attr_is_timeseries = [isinstance(data.features[i][1], TIMESERIES) for i in range(len(data.features))]
        attr_dats = self._split_attributes(data)
        self._transformer_list = []

        if self.n_jobs == 1:
            self._transformer_list = [transformer for transformer in starmap(_single_fit, [(attribute_dataset, self._attr_is_timeseries[attr_i], self.transformer, label)
                                                                                           for attr_i, attribute_dataset in enumerate(attr_dats)])]
        else:
            with Pool(self.n_jobs) as pool:
                self._transformer_list = [transformer for transformer in pool.starmap(_single_fit, [(attribute_dataset, self._attr_is_timeseries[attr_i], self.transformer, label)
                                                                                                    for attr_i, attribute_dataset in enumerate(attr_dats)])]

        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        """This method transforms the specified features of a dataset according to the specified transformer in this UniToMultivariateWrapper object.
        The resulting transformed TIMESERIES features are concatenated back in order,
        while the non-transformed non-TIMESERIES features are kept the same.

        :param data: Dataset to transform via this UniToMultivariateWrapper
        :type data: Dataset
        :return: Dataset with transformed TIMESERIES features
        :rtype: Dataset
        """
        attr_dats = self._split_attributes(data)

        transformed_attr_dats = []

        if self.n_jobs == 1:
            transformed_attr_dats = [dataset for dataset in starmap(_single_transform, [(attribute_dataset, self._attr_is_timeseries[attr_i], self._transformer_list[attr_i])
                                                                                        for attr_i, attribute_dataset in enumerate(attr_dats)])]
        else:
            with Pool(self.n_jobs) as pool:
                transformed_attr_dats = [dataset for dataset in pool.starmap(_single_transform, [(attribute_dataset, self._attr_is_timeseries[attr_i], self._transformer_list[attr_i])
                                                                                                 for attr_i, attribute_dataset in enumerate(attr_dats)])]

        # aggregate the transformed and not transformed attributes to a single dataset
        return self._aggregate_attribute_datasets(data, transformed_attr_dats)

    def _aggregate_attribute_datasets(self, original_dat, mixed_transformed_dats) -> Dataset:
        ret_dat = copy(original_dat)
        ret_dat.data = [[feature for instance in mixed_transformed_dats for feature in instance.data[itx]] for itx, _ in enumerate(original_dat.data)]
        ret_dat.features = [feature_pair for attribute_dat in mixed_transformed_dats for feature_pair in attribute_dat.features]
        return ret_dat
