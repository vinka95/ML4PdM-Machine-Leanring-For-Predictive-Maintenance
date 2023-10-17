from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from ml4pdm.evaluation import Evaluator
from ml4pdm.evaluation.metrics import loss_asymmetric, loss_false_negative_rate, loss_false_positive_rate, score_performance
from ml4pdm.parsing import DatasetParser
from ml4pdm.prediction import DirectRulApproach, SVREstimator, WindowedPredictor
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

    # Compute instance lengths for training set, which will be used later on to generate training labels for windowed dataset.
    train_instance_lengths = WindowedPredictor.extract_instance_lengths(train_dataset)
    train_dataset.target = train_instance_lengths

    model = DirectRulApproach(15, MinMaxScaler(), SVREstimator(15))

    evaluator = Evaluator(None, [model], None, [loss_asymmetric, mean_squared_error, score_performance, mean_absolute_error,
                                                mean_absolute_percentage_error, loss_false_positive_rate, loss_false_negative_rate])

    results = evaluator.evaluate_train_test_split(train_dataset, test_dataset)[0]

    for score, expected in zip(results, [11374.13, 1139.64, 0.28, 26.07, 0.60, 0.44, 0.28]):
        assert expected - 0.01 <= score <= expected + 0.01
