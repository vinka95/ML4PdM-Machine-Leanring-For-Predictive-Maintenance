
from copy import copy

import numpy as np
from pyts.preprocessing import InterpolationImputer

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.transformation import TimeSeriesTransformer, attach_timesteps


class TimeSeriesImputer(TimeSeriesTransformer):
    """This class has fit and transform methods that change the lengths of all time series instances to be equal
    and fills the resulting missing values according to a specified strategy.
    """
    class _MeanByStep:
        def __init__(self):
            self.step_mean = None

        def fit(self, x):
            self.step_mean = np.nanmean(x, axis=0)
            return self

        def transform(self, x):
            return np.asarray([[self.step_mean[i] if np.isnan(value) else value for i, value in enumerate(instance)] for instance in x])

    class _FullMean:
        def __init__(self):
            self.mean = None

        def fit(self, x):
            self.mean = np.nanmean(x)
            return self

        def transform(self, x):
            return np.asarray([[self.mean if np.isnan(value) else value for i, value in enumerate(instance)] for instance in x])

    def __init__(self, fill_strategy='mean-by-step'):
        """the constructor consists of specifying the filling strategy. The following filling strategies are supported:
        'mean-by-step' : fills the missing values of a time step by the mean of all instances which have values on that time step.
        'full-mean' : fills the missing values by the mean value of all recorded values in the dataset.
        'interpolation-[?]' where [?] has to be replaced by the strategy string for the Pyts Interpolation class.
        For further details on this option visit the following site:
        https://pyts.readthedocs.io/en/stable/generated/pyts.preprocessing.InterpolationImputer.html#pyts.preprocessing.InterpolationImputer

        :param fill_strategy: the strategy for filling the missing values, defaults to 'mean-by-step'
        :type fill_strategy: str, optional
        """
        self.fill_strategy = fill_strategy
        self.indices = None
        if self.fill_strategy == 'mean-by-step':
            self.imputer = self._MeanByStep()
        if self.fill_strategy == 'full-mean':
            self.imputer = self._FullMean()
        if self.fill_strategy.split('-')[0] == 'interpolation':
            self.imputer = InterpolationImputer(strategy=self.fill_strategy.split('-')[1])

    def _nan_fill_fit(self, x):
        self.indices = {pair[0] for instance in x for pair in instance}
        return self

    def _nan_fill_transform(self, x):
        indices_by_instance = [{pair[0] for pair in instance} for instance in x]
        return np.asarray([[dict(x[i])[index] if index in indices_by_instance[i] else np.nan for index in self.indices] for i in range(len(x))])

    def fit(self, data: Dataset, label=None, **kwargs) -> "TimeSeriesImputer":
        """This method fits the imputer algorithm to the specified data and labels. The labels are ignored for all algorithms.
        The lengths of the time series get adjusted to be the same length, which is set to the longest time series found in the specified training set.
        The original data of the time series are aligned at the end and the missing data points filled with NaN values.
        The specified fill-strategy algorithm is then fitted to fill those kinds of time series.

        :param data:  feature instances for fitting the fill strategy algorithm
        :type data: Dataset
        :param y: target instances that are ignored by the supported pyts algorithms, defaults to None
        :type y: arbitrary, optional
        :return: self
        :rtype: TimeSeriesImputer
        """
        stripped_instances = [instance[0] for instance in data.data]
        self._nan_fill_fit(stripped_instances)
        x_naned = self._nan_fill_transform(stripped_instances)
        self.imputer.fit(x_naned)
        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        """This method uses the imputer algorithm specified by the filling strategy in the constructor to impute and fix the lengths of the time series in the specified data x.
        The lengths of the time series get adjusted to be the same length.
        The length is set to the longest time series found in the training set of the data this class was fit on previously.
        The original data of the time series are aligned at the end and the missing data points filled with NaN values.
        These NaN values are filled using the fitted filling algorithm.

        :param data: Time series instances to be adjusted and imputed
        :type data: Dataset
        :return: Imputed time series instances that have the same length
        :rtype: Dataset
        """
        stripped_instances = [instance[0] for instance in data.data]
        x_naned = self._nan_fill_transform(stripped_instances)
        x_transformed = self.imputer.transform(x_naned)
        x_wrapped = [[instance] for instance in x_transformed]
        transformed_dataset = copy(data)
        transformed_dataset.data = attach_timesteps(x_wrapped, list(self.indices))
        transformed_dataset.features = [('imputed('+data.features[0][0]+')', TIMESERIES(NUMERIC(), NUMERIC()))]
        return transformed_dataset
