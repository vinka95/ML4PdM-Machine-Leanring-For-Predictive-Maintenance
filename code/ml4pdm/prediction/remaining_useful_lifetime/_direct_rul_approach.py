from sklearn.pipeline import make_pipeline

from ml4pdm.data import Dataset
from ml4pdm.prediction import RemainingUsefulLifetimeEstimator, WindowedPredictor
from ml4pdm.transformation import SklearnWrapper, TSFreshFeatureCalculators, TSFreshWrapper, UniToMultivariateWrapper, WindowingApproach


class DirectRulApproach(RemainingUsefulLifetimeEstimator):
    """This pipeline element can be used to represent the full Embed RUL approach that was described in the following paper:
    R. Khelif, B. Chebel-Morello, S. Malinowski, E. Laajili, F. Fnaiech and N. Zerhouni, 
    "Direct Remaining Useful Life Estimation Based on Support Vector Regression," 
    in IEEE Transactions on Industrial Electronics, vol. 64, no. 3, pp. 2276-2285, March 2017, doi: 10.1109/TIE.2016.2623260.
    """

    def __init__(self, window_size: int, scaler, rul_estimator: RemainingUsefulLifetimeEstimator) -> None:
        """Initializes the Direct RUL approach with custom pipeline steps for the different purposes.

        :param window_size: Size of windows the timeseries data going to be split into
        :type window_size: int
        :param scaler: Scaler that is used on the datasets used on Direct RUL approach
        :type scaler: Transformer
        :param rul_estimator: The final step of the pipeline that uses windowed timeseries instances to compare the test instances to and
                              return a RUL prediction
        :type rul_estimator: RemainingUsefulLifetimeEstimator
        """
        super().__init__()
        self.window_size = window_size
        self.offset = window_size
        self.pipeline = make_pipeline(SklearnWrapper(scaler, SklearnWrapper.extract_timeseries_concatenated, SklearnWrapper.rebuild_timeseries_concatenated), WindowedPredictor(
            WindowingApproach(self.window_size, self.offset, rul_target_calculate=False), [UniToMultivariateWrapper(TSFreshWrapper([TSFreshFeatureCalculators.MEAN]))], rul_estimator))

    def fit(self, data: Dataset, label=None, **kwargs):
        """Fits the Direct RUL approach by fitting all the different pipeline elements.
        See the above mentioned paper for more details.

        :param data: Dataset that is used to train the Direct RUL approach
        :type data: Dataset
        :return: self
        :rtype: DirectRULApproach
        """
        return self.pipeline.fit(data, label, **kwargs)

    def predict(self, data: Dataset, **kwargs):
        """Predicts the remaining useful lifetime for the machines in the dataset using the Direct RUL approach.
        See the above mentioned paper for more details.

        :param data: Dataset that the RUL predictions should be made for using Direct RUL approach
        :type data: Dataset
        :return: self
        :rtype: DirectRULApproach
        """
        return self.pipeline.predict(data, **kwargs)
