from ml4pdm.data import NUMERIC, TIMESERIES
from ml4pdm.transformation import AttributeFilter

from .uni_to_multivariate_test import generate_mixed_dataset


def test_attribute_filter():
    dataset = generate_mixed_dataset()
    filter_obj = AttributeFilter(min_unique_values=4)
    filter_obj.fit(dataset)
    filtered_dataset = filter_obj.transform(dataset)
    assert len(filtered_dataset.data[0]) == 2
    assert len(filtered_dataset.features) == 2
    assert isinstance(filtered_dataset.features[0][1], TIMESERIES)

    filter_obj = AttributeFilter(remove_indices=[0])
    filter_obj.fit(dataset)
    filtered_dataset = filter_obj.transform(dataset)
    assert len(filtered_dataset.data[0]) == 2
    assert len(filtered_dataset.features) == 2
    assert isinstance(filtered_dataset.features[0][1], NUMERIC)

    filter_obj = AttributeFilter(remove_indices=[0], min_unique_values=4)
    filter_obj.fit(dataset)
    filtered_dataset = filter_obj.transform(dataset)
    assert len(filtered_dataset.data[0]) == 1
    assert len(filtered_dataset.features) == 1
    assert isinstance(filtered_dataset.features[0][1], TIMESERIES)
