from ml4pdm.parsing import DatasetParser
from ml4pdm.data import Dataset
from ml4pdm.transformation import MovingWeightedAverage

def test_moving_weighted_average():
    
    dataset = DatasetParser.get_cmapss_data()   
    
    # Test for 10 instances.
    dataset.data = dataset.data[0:10]          

    moving_weighted_average = MovingWeightedAverage(span=5)

    transformed_data = moving_weighted_average.transform(dataset)      
    
    assert isinstance(transformed_data, Dataset)

    assert transformed_data is not None

    assert transformed_data.data is not None

    assert transformed_data.data.shape == (2136, 3)

    assert transformed_data.features is not None

    assert len(transformed_data.features) == 3

    assert isinstance(transformed_data.features[0], tuple)
 