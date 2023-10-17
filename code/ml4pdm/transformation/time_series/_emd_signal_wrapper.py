from copy import copy

import numpy as np
from PyEMD import CEEMDAN, EEMD, EMD

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.transformation import TimeSeriesTransformer, attach_timesteps


class EMDSignalWrapper(TimeSeriesTransformer):
    """This class wraps the EMD-Signal library. EMD is Empirical Mode Decomposition which is a decomposition algorithm that creates a list of IMF components.
    IMFs are intrinstic mode functions and can be interpreted as a form of orthogonal basis for the original signal.
    The EMD-Signal library consists of several different algorithms which can be specified in the constructor of this class.
    For further information on the algorithms and the argument-options for these algorithms visit https://pyemd.readthedocs.io/en/latest/
    """

    def __init__(self, emd_algorithm="emd", emd_arguments=None, keep_components=None):
        """the constructor consists of specifying the option for the empirical mode decomposition algorithm, its constructor parameters
        and also a specification for what imf-component to keep.
        There are three options for the emd_mode argument:
        "emd" - Empirical Mode Decomposition
        "eemd" - Ensemble empirical mode decomposition
        "ceemdan" - Complete ensemble EMD with adaptive noise

        : param emd_algorithm : the PyEMD algorithm to be used to generate the imf components, options: "emd", "eemd", "ceemdan". default-value = "emd"
        : type emd_algorithm : str
        : param emd_arguments : the PyEMD algorithm constructor parameters.
        : type emd_arguments : dict of (str to object)
        : param keep_components : specification of what imf components to keep. These can be a positive integers indicating the index of the imf component
        or a negative integers indicating the index counting from the back. e.g. -1 is the last imf component. 0 is the first imf component.
        You can also just specify an integer which results in keeping just the imf component at that index.  default-value = [-1]
        : type keep_components : array-like or integer

        """
        self.emd_algorithm = emd_algorithm
        self.emd_arguments = emd_arguments
        self.keep_components = keep_components
        if emd_algorithm == "emd":
            if emd_arguments is None:
                self.decompositor = EMD()
            else:
                self.decompositor = EMD(**emd_arguments)
        elif emd_algorithm == "eemd":
            if emd_arguments is None:
                self.decompositor = EEMD()
            else:
                self.decompositor = EEMD(**emd_arguments)
        elif emd_algorithm == "ceemdan":
            if emd_arguments is None:
                self.decompositor = CEEMDAN()
            else:
                self.decompositor = CEEMDAN(**emd_arguments)
        if keep_components is None:
            keep_components = [-1]
        if hasattr(keep_components, '__len__'):
            self.keep_imfs = keep_components
        else:
            self.keep_imfs = [keep_components]

    def fit(self, data: Dataset, label=None, **kwargs) -> "EMDSignalWrapper":
        """Does nothing but returns this object

        :param data: Dataset with instances to be decomposed
        :type data: Dataset
        :param label: Ignored, defaults to None
        :type label: Arbitrary, optional
        :return: Self
        :rtype: EMDSignalWrapper
        """
        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        """Decomposes each time series in the specified dataset into the imf components.
        It selects only the one component at index keep_imf specified in the constructor of this object.

        :param data: Time series instances instances to be decomposed
        :type data: Dataset
        :return: Data set with instances consisting of imf components
        :rtype: Dataset
        """
        data_lists = data.get_time_series_data_as_array()
        x_transformed = []
        for instance in data_lists:
            ts_feature = instance[0]
            deco = self.decompositor(ts_feature)
            ts_len = len(ts_feature)
            instance_ret = []
            for idx in self.keep_imfs:
                num_deco = len(deco)
                if -num_deco <= idx < num_deco:
                    instance_ret.append(deco[idx])
                else:
                    instance_ret.append(np.zeros(ts_len))
            x_transformed.append(np.asarray(instance_ret))

        transformed_dataset = copy(data)
        transformed_dataset.data = attach_timesteps(x_transformed)
        transformed_dataset.features = [('emd['+str(idx)+']('+data.features[0][0]+')', TIMESERIES(NUMERIC(), NUMERIC())) for idx in self.keep_imfs]
        return transformed_dataset
