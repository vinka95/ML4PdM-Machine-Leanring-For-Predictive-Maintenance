from ml4pdm.prediction import Predictor
from ml4pdm.transformation import TimeSeriesTransformer


class HealthIndexEstimator(Predictor, TimeSeriesTransformer):
    """Abstract base class for Health Index Regressors. They are Predictor and TimeSeriesTransformer at the same time and support both modes.
    """
