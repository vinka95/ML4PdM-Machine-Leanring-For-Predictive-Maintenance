from math import exp
from typing import List

import numpy as np

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.prediction import RemainingUsefulLifetimeEstimator


class RULEstimator(RemainingUsefulLifetimeEstimator):
    """The RULEstimator is used to predict the remaining useful lifetime of a machine given its health index curve.
    The health index curve is matched as best as possible with other hi curves that were recorded until failure. For this purpose
    the HI curve is moved by an offset limited by max_time_lag to find the best matching curve. This way the remaining time
    until breakdown can be computed by looking at the time difference between the last timesteps of both hi curves once the match is
    near perfect.

    The following methods implement Formula 6 and 7 of the following paper:
    Narendhar Gugulothu, Vishnu TV, Pankaj Malhotra, et al. „Predicting Remaining
    Useful Life using Time Series Embeddings based on Recurrent Neural Networks“.
    In: CoRR abs/1709.01073 (2017). arXiv: 1709.01073
    """

    def __init__(self, window_size: int, lambda_param: float, max_time_lag: int, alpha: float, max_rul: int) -> None:
        """Initializes the RULEstimator with the important parameters for calculating the RUL of a machine.

        :param window_size: Window size that was used to transform the Dataset
        :type window_size: int
        :param lambda_param: Defines the factor by which the HI curve difference is divided and therefore controls the notion of similarity
        :type lambda_param: float
        :param max_time_lag: The maximum amount that the HI curve can be moved while trying to find a good match
        :type max_time_lag: int
        :param alpha: Defines the percentage of the maximum similarity that is used as a threshold. Only HI curves with a greater or equal
                      similarity will be considered for RUL computation
        :type alpha: float
        :param max_rul: Limits the maximum RUL prediction to prevent predictions that exceed the total lifetime of a machine
        :type max_rul: int
        """
        super().__init__()
        self.health_indices = []
        self.window_size = window_size
        self.lambda_param = lambda_param
        self.max_time_lag = max_time_lag
        self.alpha = alpha
        self.max_rul = max_rul

    def fit(self, data: Dataset, label=None, **kwargs) -> "RULEstimator":
        """Stores the health index curves from the training set which need to be run-to-failure. These will be used for comparisons later
        and the RUL will be predicted based on them.

        :param data: Dataset containing all run-to-failure HI curves
        :type data: Dataset
        :return: Self
        :rtype: RULEstimator
        """
        _, hi_curves = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        self.health_indices.extend([[features[0] for features in hi_curve] for hi_curve in hi_curves])
        return self

    def similarity(self, test_inst: List[float], train_inst: List[float], time_lag: int) -> float:
        """Computes the similarity of two HI curves by computing the element wise difference which is then used in 
        an exponential function with some additional factors. See Formula 6 in the above mentioned paper.

        :param test_inst: HI curve that the RUL should be predicted for
        :type test_inst: List[float]
        :param train_inst: One of the normal HI curves that is run to failure
        :type train_inst: List[float]
        :param time_lag: Amount that the test HI curve was moved to best match the normal HI curve
        :type time_lag: int
        :return: Returns the value of similarity for these two HI curves
        :rtype: float
        """
        diff = 0.0
        for k, inst in enumerate(test_inst):
            diff += (inst-train_inst[k+time_lag])**2/self.lambda_param
        return exp((-1.0/(len(test_inst)+self.window_size-1))*diff)

    def estimate(self, test_inst: List[float], train_inst: List[float], time_lag: int) -> int:
        """Computes a RUL estimate for one of the normal HI curves. The difference in length subtracted by the amount that the test HI curve
        has been moved is returned. This is because the normal HI curve is run to failure. See Figure 1 in the above mentioned paper.

        :param test_inst: HI curve that the RUL should be predicted for
        :type test_inst: List[float]
        :param train_inst: One of the normal HI curves that is run to failure
        :type train_inst: List[float]
        :param time_lag: Amount that the test HI curve was moved to best match the normal HI curve
        :type time_lag: int
        :return: Returns a RUL estimate for the test HI curve using this particular normal HI curve
        :rtype: int
        """
        return len(train_inst)-len(test_inst)-time_lag

    def rul(self, test_inst: List[float]) -> float:
        """Computes the remaining useful lifetime using the given HI curve. Computes a weighted average of the RUL predictions
        when comparing the test HI curve with the normal HI curves. Only predictions are considered where the similarity is greater or equal to
        the threshold given by alpha multiplied with the maximum similarity. See Formula 7 in the above mentioned paper.

        :param test_inst: HI curve that the RUL should be predicted for
        :type test_inst: List[float]
        :return: RUL prediction for the given HI curve
        :rtype: float
        """
        similarities = []
        for i in range(len(self.health_indices)):
            for time_diff in range(1, self.max_time_lag+1):
                if time_diff+len(test_inst) > len(self.health_indices[i]):
                    continue
                similarities.append((i, time_diff, self.similarity(test_inst, self.health_indices[i], time_diff)))
        rul = 0.0
        if len(similarities) == 0:
            return rul
        s_max = np.max([x[2] for x in similarities])
        similarities = list(filter(lambda x: x[2] >= self.alpha*s_max, similarities))
        for sim in similarities:
            rul += sim[2]*self.estimate(test_inst, self.health_indices[sim[0]], sim[1])
        rul = rul / sum([x[2] for x in similarities])
        return min(rul, self.max_rul)

    def predict(self, data: Dataset, **kwargs) -> List[float]:
        """Predicts the remaining useful lifetime for a set of health index curves by comparing them to Hi curves that were run to failure.

        :param data: Dataset of HI curves to predict the RUL for
        :type data: Dataset
        :return: RUL predictions for every instance in the dataset
        :rtype: List[float]
        """
        _, hi_curves = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        return list(map(self.rul, [[features[0] for features in hi_curve] for hi_curve in hi_curves]))
