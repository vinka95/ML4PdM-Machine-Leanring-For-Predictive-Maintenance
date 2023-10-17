from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from ml4pdm.evaluation import Evaluator
from ml4pdm.evaluation.metrics import loss_asymmetric, loss_false_negative_rate, loss_false_positive_rate, score_performance
from ml4pdm.parsing import DatasetParser
from ml4pdm.prediction import MultipleClassifierApproach
from ml4pdm.transformation import SklearnWrapper


def test_fit_predict():
    train_dataset, test_dataset = DatasetParser.get_cmapss_data(test=True)
    train_dataset.data = train_dataset.data[:30]
    test_dataset.data = test_dataset.data[:25]
    test_dataset.target = test_dataset.target[:25]

    mca = make_pipeline(SklearnWrapper(MinMaxScaler(), SklearnWrapper.extract_timeseries_concatenated,
                                       SklearnWrapper.rebuild_timeseries_concatenated),
                        MultipleClassifierApproach(range(30, 151, 30), KNeighborsClassifier))

    evaluator = Evaluator(None, [mca], None, [loss_asymmetric, mean_squared_error, score_performance, mean_absolute_error,
                                              mean_absolute_percentage_error, loss_false_positive_rate, loss_false_negative_rate])

    results = evaluator.evaluate_train_test_split(train_dataset, test_dataset)[0]

    for score, expected in zip(results, [1244.04, 1548.92, 0.08, 35.48, 0.44, 0.84, 0.08]):
        assert expected - 0.01 <= score <= expected + 0.01
