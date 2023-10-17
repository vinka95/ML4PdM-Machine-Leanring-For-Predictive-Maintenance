
import warnings
from copy import copy
from enum import Enum

import numpy as np
import scipy
from pyts.transformation import BOSS, ROCKET, WEASEL, BagOfPatterns, ShapeletTransform

from ml4pdm.data import NUMERIC, Dataset
from ml4pdm.transformation import FixedSizeFeatureExtractor


class PytsSupportedAlgorithm(Enum):
    """This enum contains as keywords all the supported transformation algorithms in pyts.
    These are the following:
    BOSS = Bag-of-SFA Symbols, which goes into the frequency domain and uses a bag-of-words algorithm to then bin those results with multiple-coefficient-binning
    ROCKET = RandOm Convolutional KErnel Transform, which calculates randomized kernels over the time series
    SHAPELET = Shapelet Transform, which calculates three kinds of similarities: time, change and shape.
    BOP = Bag of Patterns, which calculates simple pattern distributions over the data and represents them in histograms.
    and WEASEL = Word ExtrAction for time SEries cLassification, which uses several different steps including windowing, supervised symbolic representation and bag-of-patterns
    More details can be found at https://pyts.readthedocs.io/en/stable/auto_examples/index.html#transformation-algorithms
    """
    BOSS = BOSS
    ROCKET = ROCKET
    SHAPELET = ShapeletTransform
    BOP = BagOfPatterns
    WEASEL = WEASEL

    def generate_transformer(self, params=None):
        """This method generates an object of the supported algorithms class with the specified params. If params is None, the default values for the constructor are used.

        : param params: constructor params for each supported algorithm class
        : type params: dict of(string to object)
        """
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            if params is None:
                return self.value()
            return self.value(**params)


class PytsTransformWrapper(FixedSizeFeatureExtractor):
    """A wrapper for the supported pyts transformation algorithms. The supported algorithms are specified in the PytsSupportedAlgorithm enum.
    The fit and the transform methods change the AttributeTypes from TIMESERIES to a list of NUMERIC.
    Datasets with differing time series features are not supported.
    """

    def __init__(self, algorithm, algorithm_params=None):
        """Constructs the wrapper by specifying the algorithm to be used and its constructor parameters.

        :param algorithm: The pyts algorithm that is used for transformation
        :type algorithm:  PytsSupportedAlgorithm
        :param algorithm_params: dictionary of parameter specifications for the pyts transformation algorithm class constructor, defaults to None
        :type algorithm_params: dict of (string to object), optional
        :raises AttributeError: If the specified algorithm is not supported
        """
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params

        if not isinstance(algorithm, PytsSupportedAlgorithm):
            raise AttributeError
        self.transformer = self.algorithm.generate_transformer(self.algorithm_params)

    def _timeseries_shaper(self, data: Dataset):
        ts_data = data.get_time_series_data_as_array()
        stripped_ts_data = np.asarray([np.asarray(instance[0]) for instance in ts_data])
        if len(np.unique([len(instance) for instance in stripped_ts_data])) > 1:
            raise TypeError("Datasets with differing lengths of time series are not supported.")

        return stripped_ts_data

    def fit(self, data: Dataset, label=None, **kwargs):
        """This method fits the pyts algorithm to the specified data and labels. The labels are ignored for most algorithms.
        The specified data gets shaped into a format which is supported by the pyts library.
        If the specified time series data are of differing lengths, a TypeError is raised as this is not supported in the pyts library.
        Make sure to fix the lengths of the time series in order for this wrapper to work.

        :param data: Dataset to fit this wrapper on
        :type data: Dataset
        :param label: ignored, because the label/target is taken from data.target, defaults to None
        :type label: arbitrary, optional
        :return: Self
        :rtype: PytsTransformWrapper
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.transformer.fit(self._timeseries_shaper(data), data.target)

        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        """This method transforms the specified feature instances using the fitted transformation algorithm, which was specified in the constructor.

        :param data: feature instances to be transformed
        :type data: list of feature instances
        :return: feature instances that are transformed by the pyts algorithm specified in the constructor.
        :rtype: list of feature instances

        :param data: Dataset to be transformed
        :type data: Dataset
        :return: Transformed Dataset
        :rtype: Dataset
        """
        x_transformed = self.transformer.transform(self._timeseries_shaper(data))
        dataset_transformed = copy(data)
        if isinstance(x_transformed, scipy.sparse.csr.csr_matrix):
            x_transformed = x_transformed.toarray()
        dataset_transformed.data = x_transformed
        dataset_transformed.features = [('pyts['+str(i)+']('+data.features[0][0]+')', NUMERIC()) for i in range(len(x_transformed[0]))]
        return dataset_transformed
