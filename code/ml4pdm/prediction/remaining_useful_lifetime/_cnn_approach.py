from sklearn.pipeline import make_pipeline

from ml4pdm.data import Dataset
from ml4pdm.prediction import RemainingUsefulLifetimeEstimator
from ml4pdm.transformation import SklearnWrapper, WindowingApproach


class CNNRulApproach(RemainingUsefulLifetimeEstimator):
    """A novel deep Convolutional Neural Network (CNN) based regression approach for estimating the RUL is implemented here,
    as explained in the following paper:

    G. Sateesh Babu, Peilin Zhao, and Xiao-Li Li 
    'Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life' 
    Institute for Infocomm Research, A  STAR, Singapore
    """

    def __init__(self, scaler, window_size: int, rul_estimator: RemainingUsefulLifetimeEstimator) -> None:
        """Initializes the CNN based approach with custom pipeline steps for the different purposes.

        :param scaler1: Scaler that is used on the datasets used as part of CNN RUL approach
        :type scaler1: Transformer
        :param scaler2: Scaler that is used on the datasets used as part of CNN RUL approach
        :type scaler2: Transformer
        :param window_size: Size of windows the timeseries data going to be split into
        :type window_size: int
        :param rul_estimator: The final step of the pipeline that uses windowed timeseries instances to compare the test instances to and
                              return a RUL prediction
        :type rul_estimator: RemainingUsefulLifetimeEstimator
        """
        super().__init__()
        self.window_size = window_size
        self.scaler = scaler
        self.pipeline = make_pipeline(SklearnWrapper(self.scaler, SklearnWrapper.extract_timeseries_concatenated, SklearnWrapper.rebuild_timeseries_concatenated),
                                      WindowingApproach(window_size=self.window_size, offset=self.window_size), rul_estimator)

    def fit(self, data: Dataset, label=None, **kwargs):
        """Fits data using CNN RUL approach through all the different pipeline elements.
        See the above mentioned paper for more details

        :param data: Dataset that is used to train the model with.
        :type data: Dataset
        :param label: Target labels, defaults to None
        :type label: [type], optional
        :return: self
        :rtype: CNNRulApproach
        """
        return self.pipeline.fit(data, label, **kwargs)

    def predict(self, data: Dataset, **kwargs):
        """Predicts the remaining useful lifetime for the machines in the dataset using the CNN model.
        See the above mentioned paper for more details.

        :param data: Dataset that the RUL predictions should be made for using Direct RUL approach
        :type data: Dataset
        :return: self
        :rtype: CNNRulApproach
        """
        return self.pipeline.predict(data, **kwargs)
