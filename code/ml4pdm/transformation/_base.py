
import numpy as np


def listify_time_series(data):
    """This method transforms the dataset.data of univariate timeseries into a list format for further processing.
    Specifically time series in the tuple (time-stamp, value) pair format are transformed into a simple list,
    where the time-stamp information is lost.

    : param data: data set with univariate time series instances in list or tuple format
    : type data: list of time series
    :return: data set with univariate time series instances in list format
    :rtype: list of (list of float)
    """
    if isinstance(data[0][0], tuple):
        return [np.asarray([pair[1] for pair in instance]) for instance in data]
    return [np.asarray(instance) for instance in data]


def attach_timesteps(data, timesteps=None):
    """attaches timesteps to each timeseries to create (timestep, value) pairs.
    If timesteps is None, the default (0 to len(timeseries)) timesteps are attached.
    This function requires that all features of the specified instances (data) are timeseries.

    :param data: The instances consisting of timeseries features as simple value arrays
    :type data: List[List[Instance]]
    :param timesteps: The list of timesteps, defaults to None
    :type timesteps: List[int]
    :return: The Instances with the same timeseries features but with attached timesteps
    :rtype: List[List[List[Tuple]]]
    """
    if timesteps is None:
        return [[[(timestep, value) for timestep, value in enumerate(timeseries)] for timeseries in instance] for instance in data]
    return [[[(timesteps[i], value) for i, value in enumerate(timeseries)] for timeseries in instance] for instance in data]
