
from copy import copy

import numpy as np
import pywt

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.transformation import TimeSeriesTransformer, attach_timesteps


class PywtWrapper(TimeSeriesTransformer):
    """This class wraps the pywt library https://pywavelets.readthedocs.io/en/latest/.
    This library is used to calculate the discrete wavelet transformation,
    which is a time and frequency scanning approach that uses wavelets as a filter.
    A scan consists of multiplying the wavelet translated at several different points.
    Wavelets are specific short waves which satisfy certain conditions.
    E.g. they trail on both sides to 0, such that we retain finite values while scanning.
    This generates two timeseries, an approximation and a detail timeseries.
    """

    def __init__(self, wavelet='db1', mode='symmetric', interleaving=False):
        """Initializes the PywtWrapper.
        For options for the wavelet argument use wavelist() or https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
        For options for the mode argument use modelist() or https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes
        interleaving==True means that the resulting approximantion and detail timeseries are combined into one.
        If that is False, we get two timeseries features.

        :param wavelet: the wavelet to be used for scanning, defaults to 'db1'
        :type wavelet: str, optional
        :param mode: signal extension mode, defaults to 'symmetric'
        :type mode: str, optional
        :param interleaving: decides wether or not to interleave the resulting timeseries, defaults to False
        :type interleaving: bool, optional
        """
        self.interleaving = interleaving
        self.wavelet = wavelet
        self.mode = mode

    @staticmethod
    def wavelist():
        """Getter for the wavelet options

        :return: List of wavelet options as strings
        :rtype: List[str]
        """
        return pywt.wavelist(kind='discrete')

    @staticmethod
    def modelist():
        """Getter for the mode options

        :return: Lis of mode options as strings
        :rtype: List[str]
        """
        return pywt.Modes.modes

    def fit(self, data: Dataset, label=None, **kwargs) -> "PywtWrapper":
        """Does nothing but returns self

        :param data: will be ignored
        :type data: Dataset
        :param label: will be ignored defaults to None
        :type label: arbitrary, optional
        :return: self
        :rtype: PywtWrapper
        """
        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        """Transforms the instances of the specified dataset according to the specifications in the constructor.

        :param data: Dataset with instances to be transformed
        :type data: Dataset
        :return: Dataset with transformed instances by this pywt wrapper
        :rtype: Dataset
        """
        data_lists = data.get_time_series_data_as_array()
        x_transformed = []

        for instance in data_lists:
            approx, detail = pywt.dwt(data=instance[0], wavelet=self.wavelet, mode=self.mode)
            if self.interleaving:
                x_transformed.append(np.asarray([[value for pair in zip(approx, detail) for value in pair]]))
            else:
                x_transformed.append(np.asarray([approx, detail]))

        transformed_dataset = copy(data)
        transformed_dataset.data = attach_timesteps(x_transformed)
        feature_label = data.features[0][0]
        if self.interleaving:
            transformed_dataset.features = [('pywavelet('+feature_label+')', TIMESERIES(NUMERIC(), NUMERIC()))]
        else:
            transformed_dataset.features = [('pywt-approx('+feature_label+')', TIMESERIES(NUMERIC(), NUMERIC())),
                                            ('pywt-detail('+feature_label+')', TIMESERIES(NUMERIC(), NUMERIC()))]

        return transformed_dataset
