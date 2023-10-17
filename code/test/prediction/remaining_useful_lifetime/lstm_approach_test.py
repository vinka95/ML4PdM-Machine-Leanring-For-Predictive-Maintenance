import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.evaluation import Evaluator
from ml4pdm.evaluation.metrics import loss_asymmetric, loss_false_negative_rate, loss_false_positive_rate, score_performance
from ml4pdm.parsing import DatasetParser
from ml4pdm.prediction import LSTMEstimator, LSTMRulApproach
from ml4pdm.transformation import AttributeFilter


def test_fit_predict():
    train_dataset, test_dataset = DatasetParser.get_cmapss_data(test=True)

    # Train - Attribute filter - remove features with constant or nan values.
    filter_obj = AttributeFilter(min_unique_values=3)
    filter_obj.fit(train_dataset)
    train_dataset = filter_obj.transform(train_dataset)

    # Test - Attribute filter - remove features with constant or nan values.
    filter_obj2 = AttributeFilter(min_unique_values=3)
    filter_obj2.fit(test_dataset)
    test_dataset = filter_obj2.transform(test_dataset)

    scaler1 = StandardScaler()
    scaler2 = MinMaxScaler()

    pipeline = LSTMRulApproach(scaler1, scaler2, 15, LSTMEstimator(15, 17))

    pipeline.fit(train_dataset)

    results = pipeline.predict(test_dataset)

    assert(len(test_dataset.target) == len(results))
