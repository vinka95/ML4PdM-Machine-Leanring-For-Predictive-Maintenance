from ml4pdm.transformation import TSFreshFeatureCalculators, TSFreshWrapper
import pytest
from ml4pdm.parsing import DatasetParser
from ml4pdm.data import Dataset

def _get_wrong_enum():  
    wrong_features = [
        TSFreshFeatureCalculators.ABSOLUTE_SUM_OF_CHANGES,
        "agg_autocorrelation"
        ]

    return wrong_features

def _get_correct_enum():
    correct_features = [
        TSFreshFeatureCalculators.ABSOLUTE_SUM_OF_CHANGES,
        TSFreshFeatureCalculators.MAXIMUM,
        TSFreshFeatureCalculators.MEDIAN,
        TSFreshFeatureCalculators.KURTOSIS,
        TSFreshFeatureCalculators.LARGE_STANDARD_DEVIATION        
        ]

    return correct_features

def test_constructor_features_name():
    with pytest.raises(AttributeError):
        TSFreshWrapper(features_name=_get_wrong_enum())

def test_tsfresh_wrapper():
    dataset = DatasetParser.get_cmapss_data()
    dataset.data = dataset.data[0:10]   

    # Calculate a comprehensive set of features from tsfresh.
    wrapper = TSFreshWrapper()
    all_features = wrapper.transform(x=dataset)
    # All features -> Extraction for 10 instances is around 30 secondes.
    # All features -> Extraction for 100 instances is around 4.30 minutes. 
    
    assert isinstance(all_features, Dataset)

    assert all_features is not None

    assert all_features.data.shape == (10, 18696)

    assert len(all_features.features) == 18696

    assert isinstance(all_features.features[0], tuple)

    # Calculate more of a certain type of features from tsfresh.
    wrapper = TSFreshWrapper(features_name=_get_correct_enum())
    filtered_features = wrapper.transform(x=dataset)
    # Five features -> Extraction for 10 instances is around 30 secondes.
    # Five features -> Extraction for 100 instances is around 4.30 minutes.
    
    assert isinstance(filtered_features, Dataset)  

    assert filtered_features is not None    
    
    assert filtered_features.data.shape == (10, 552)

    assert len(filtered_features.features) == 552

    assert isinstance(filtered_features.features[0], tuple)
