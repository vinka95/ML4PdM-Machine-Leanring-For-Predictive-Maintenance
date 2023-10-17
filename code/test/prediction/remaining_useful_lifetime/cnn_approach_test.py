from sklearn.preprocessing import MinMaxScaler

from ml4pdm.parsing import DatasetParser
from ml4pdm.prediction import CNNEstimator, CNNRulApproach
from ml4pdm.transformation import AttributeFilter


def test_cnn_approach():
    train_object, test_object = DatasetParser.get_cmapss_data(test=True)

    # Train - Attribute filter - remove features with constant or nan values.
    filter_obj = AttributeFilter(min_unique_values=3)
    filter_obj.fit(train_object)
    train_object = filter_obj.transform(train_object)

    # Test - Attribute filter - remove features with constant or nan values.
    filter_obj2 = AttributeFilter(min_unique_values=3)
    filter_obj2.fit(test_object)
    test_object = filter_obj2.transform(test_object)

    scaler = MinMaxScaler()

    pipeline = CNNRulApproach(scaler, 15, CNNEstimator(len(train_object.features) - 1, 15, 15))

    pipeline.fit(train_object)

    results = pipeline.predict(test_object)

    assert(len(test_object.target) == len(results))
