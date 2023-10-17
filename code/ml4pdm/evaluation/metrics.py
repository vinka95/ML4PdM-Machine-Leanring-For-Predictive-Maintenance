"""This module holds metrics that extend the ones provided in the sklearn library.
The metrics are fully compatible with sklearn.
"""
import math

from sklearn.utils.validation import check_consistent_length


def loss_lin_lin(y_true, y_pred, late_multiplier=1.0, early_multiplier=1.0):
    """This metric converts the absolute error function into an asymmetric loss function.
    This is achieved by multiplying early and late predictions with potentially differing factors.
    By default this function behaves like a standard absolute error loss.

    References:

    * Divish Rengasamy, Benjamin Rothwell, and Grazziela P. Figueredo.
      „Asymmetric Loss Functions for Deep Learning Early Predictions of Remaining Useful Life in Aerospace Gas Turbine Engines“.
      In: 2020 International Joint Conference on Neural Networks, IJCNN 2020, Glasgow, United Kingdom, July 19-24, 2020. IEEE, 2020

    :param y_true: Ground truth (correct) target values.
    :type y_true: array-like of float
    :param y_pred: Estimated target values.
    :type y_pred: array-like of float
    :param late_multiplier: The multiplier that is applied to late predictions. The higher the value, the more penalty is applied, defaults to 1.0
    :type late_multiplier: float, optional
    :param early_multiplier: The multiplier that is applied to early predictions. The higher the value, the more penalty is applied, defaults to 1.0
    :type early_multiplier: float, optional
    :return: A single floating point value. Lower is better.
    :rtype: float
    """
    check_consistent_length(y_true, y_pred)

    result = 0.0
    for y_p, y_t in zip(y_pred, y_true):
        diff = y_p - y_t
        if diff <= 0.0:
            result += -early_multiplier * diff
        else:
            result += late_multiplier * diff

    return result


def loss_asymmetric(y_true, y_pred, late_divisor=10.0, early_divisor=13.0):
    """This metric assigns a score :math:`S` to a prediction based on the difference between the
    predicted value :math:`\\hat{r}` and the real RUL :math:`r_t` as follows:

    :math:`s_n = \\begin{cases} e^{-(r_t - \\hat{r}_t)/10}-1 & \\text{if } r_t - \\hat{r}_t \\leq 0\\\\ e^{(r_t - \\hat{r}_t)/13}-1 & \\text{if } r_t - \\hat{r}_t > 0 \\end{cases}`

    References:

    * Racha Khelif, Brigitte Chebel-Morello, Simon Malinowski, et al.
      „Direct Remaining Useful Life Estimation Based on Support Vector Regression“.
      In: IEEE Trans. Ind. Electron. 64.3 (2017)

    :param y_true: Ground truth (correct) target values.
    :type y_true: array-like of float
    :param y_pred: Estimated target values.
    :type y_pred: array-like of float
    :param late_divisor: The divisor that is applied to late predictions. The higher the value, the less penalty is applied, defaults to 10.0
    :type late_divisor: float, optional
    :param early_divisor: The divisor that is applied to early predictions. The higher the value, the less penalty is applied, defaults to 13.0
    :type early_divisor: float, optional
    :return: A single floating point value. Lower is better.
    :rtype: float
    """
    check_consistent_length(y_true, y_pred)

    result = 0.0
    for y_t, y_p in zip(y_true, y_pred):
        diff = y_t - y_p
        if diff <= 0.0:
            result += math.exp(-diff / late_divisor) - 1.0
        else:
            result += math.exp(diff / early_divisor) - 1.0

    return result


def score_performance(y_true, y_pred, lower_bound=-10.0, upper_bound=13.0):
    """This metric evaluates the performance as the percentage of correct predictions.
    A prediction is considered correct if its error, defined as :math:`E =\\text{ Actual RUL }-\\text{ Predicted RUL}`,
    falls within the interval :math:`I`.
    :math:`I` defaults to :math:`[-10, 13]` which means that early predictions are more tolerable.

    References:

    * Racha Khelif, Brigitte Chebel-Morello, Simon Malinowski, et al.
      „Direct Remaining Useful Life Estimation Based on Support Vector Regression“.
      In: IEEE Trans. Ind. Electron. 64.3 (2017)

    :param y_true: Ground truth (correct) target values.
    :type y_true: array-like of float
    :param y_pred: Estimated target values.
    :type y_pred: array-like of float
    :param lower_bound: The lower bound for the difference that will still be accepted as a correct prediction, defaults to -10.0
    :type lower_bound: float, optional
    :param upper_bound: The upper bound for the difference that will still be accepted as a correct prediction, defaults to 13.0
    :type upper_bound: float, optional
    :return: A single floating point value that represents a percentage and therefore falls in the range [0.0, 1.0]
    :rtype: float
    """
    check_consistent_length(y_true, y_pred)

    count = 0.0
    for y_t, y_p in zip(y_true, y_pred):
        diff = y_t - y_p
        if diff >= lower_bound and diff <= upper_bound:
            count += 1.0

    return count / float(len(y_true))


def loss_false_positive_rate(y_true, y_pred, tolerance=-13):
    """Returns a value between 0.0 and 1.0 that resembles the percentage of predictions that were too small/early and outside of the tolerance limit.

    :param y_true: Ground truth (correct) target values
    :type y_true: array-like of float
    :param y_pred: Estimated target values
    :type y_pred: array-like of float
    :param tolerance: tolerance below which the predictions are counted as false positives, defaults to -13
    :type tolerance: int, optional
    :return: false positive rate between 0.0 and 1.0
    :rtype: float
    """
    check_consistent_length(y_true, y_pred)

    return len(list(filter(lambda x: x[1]-x[0] < tolerance, list(zip(y_true, y_pred)))))/len(y_true)


def loss_false_negative_rate(y_true, y_pred, tolerance=10):
    """Returns a value between 0.0 and 1.0 that resembles the percentage of predictions that were too large/late and outside of the tolerance limit.

    :param y_true: Ground truth (correct) target values
    :type y_true: array-like of float
    :param y_pred: Estimated target values
    :type y_pred: array-like of float
    :param tolerance: tolerance above which the predictions are counted as false positives, defaults to 10
    :type tolerance: int, optional
    :return: false negative rate between 0.0 and 1.0
    :rtype: float
    """
    check_consistent_length(y_true, y_pred)

    return len(list(filter(lambda x: x[1]-x[0] > tolerance, list(zip(y_true, y_pred)))))/len(y_true)
