from sklearn.pipeline import make_pipeline

from ml4pdm.data import Dataset
from ml4pdm.prediction import RemainingUsefulLifetimeEstimator
from ml4pdm.transformation import SklearnWrapper, WindowingApproach


class LSTMRulApproach(RemainingUsefulLifetimeEstimator):
    """A Long Short-Term Memory (LSTM) approach for RUL estimation, 
    which can make full use of the sensor sequence information and expose hidden patterns within sensor data with
    multiple operating conditions, fault and degradation models is implemented here, as explained in the following paper:

    S. Zheng, K. Ristovski, A. Farahat and C. Gupta, 
    "Long Short-Term Memory Network for Remaining Useful Life estimation," 
    2017 IEEE International Conference on Prognostics and Health Management (ICPHM), 
    2017, pp. 88-95, doi: 10.1109/ICPHM.2017.7998311.
    """

    def __init__(self, scaler1, scaler2, window_size: int, rul_estimator: RemainingUsefulLifetimeEstimator) -> None:
        """Initializes the LSTM approach with custom pipeline steps for the different purposes.

        :param scaler1: Scaler that is used on the datasets used as part of LSTM RUL approach
        :type scaler1: Transformer
        :param scaler2: Scaler that is used on the datasets used as part of LSTM RUL approach
        :type scaler2: Transformer
        :param window_size: Size of windows the timeseries data going to be split into
        :type window_size: int
        :param rul_estimator: The final step of the pipeline that uses windowed timeseries instances to compare the test instances to and
                              return a RUL prediction
        :type rul_estimator: RemainingUsefulLifetimeEstimator
        """
        super().__init__()
        self.window_size = window_size
        self.scaler1 = scaler1
        self.scaler2 = scaler2
        self.pipeline = make_pipeline(SklearnWrapper(make_pipeline(self.scaler1, self.scaler2), SklearnWrapper.extract_timeseries_concatenated, SklearnWrapper.rebuild_timeseries_concatenated),
                                      WindowingApproach(window_size=self.window_size, offset=self.window_size), rul_estimator)

    def fit(self, data: Dataset, label=None, **kwargs):
        """Fits data using LSTM RUL approach through all the different pipeline elements.
        See the above mentioned paper for more details

        :param data: Dataset that is used to train the model with.
        :type data: Dataset
        :param label: Target labels, defaults to None
        :type label: [type], optional
        :return: self
        :rtype: LSTMRulApproach
        """
        return self.pipeline.fit(data, label, **kwargs)

    def predict(self, data: Dataset, **kwargs):
        """Predicts the remaining useful lifetime for the machines in the dataset using the LSTM model.
        See the above mentioned paper for more details.

        :param data: Dataset that the RUL predictions should be made for using Direct RUL approach
        :type data: Dataset
        :return: self
        :rtype: LSTMRulApproach
        """
        return self.pipeline.predict(data, **kwargs)
