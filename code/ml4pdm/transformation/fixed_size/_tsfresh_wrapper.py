from enum import Enum
from typing import List
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import tsfresh
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.feature_extraction import extract_features
from ml4pdm.data import Dataset, NUMERIC
from ml4pdm.transformation import FixedSizeFeatureExtractor

class TSFreshFeatureCalculators(Enum):
    """This class contains the Enum as keywords which is supported by tsfresh library for feature calculators.
    There are around 70+ features are available for time series feature extraction. More information can be found at 
    https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#module-tsfresh.feature_extraction.feature_calculators.
    """
    ABS_ENERGY = "abs_energy"
    ABSOLUTE_SUM_OF_CHANGES = "absolute_sum_of_changes"
    AGG_AUTOCORRELATION = "agg_autocorrelation"
    AGG_LINEAR_TREND = "agg_linear_trend"
    APPROXIMATE_ENTROPY = "approximate_entropy"
    AR_COEFFICIENT = "ar_coefficient"
    AUGMENTED_DICKEY_FULLER = "augmented_dickey_fuller"
    AUTOCORRELATION = "autocorrelation"
    BENFORD_CORRELATION = "benford_correlation"
    BINNED_ENTROPY = "binned_entropy"
    C3 = "c3"
    CHANGE_QUANTILES = "change_quantiles"
    CID_CE = "cid_ce"
    COUNT_ABOVE = "count_above"
    COUNT_ABOVE_MEAN = "count_above_mean"
    COUNT_BELOW = "count_below"
    COUNT_BELOW_MEAN = "count_below_mean"
    CWT_COEFFICIENTS = "cwt_coefficients"
    ENERGY_RATIO_BY_CHUNKS = "energy_ratio_by_chunks"
    FFT_AGGREGATED = "fft_aggregated"
    FFT_COEFFICIENT = "fft_coefficient"
    FIRST_LOCATION_OF_MAXIMUM = "first_location_of_maximum"
    FIRST_LOCATION_OF_MINIMUM = "first_location_of_minimum"
    FOURIER_ENTROPY = "fourier_entropy"
    FRIEDRICH_COEFFICIENTS = "friedrich_coefficients"
    HAS_DUPLICATE = "has_duplicate"
    HAS_DUPLICATE_MAX = "has_duplicate_max"
    HAS_DUPLICATE_MIN = "has_duplicate_min"
    INDEX_MASS_QUANTILE = "index_mass_quantile"
    KURTOSIS = "kurtosis"
    LARGE_STANDARD_DEVIATION = "large_standard_deviation"
    LAST_LOCATION_OF_MAXIMUM = "last_location_of_maximum"
    LAST_LOCATION_OF_MINIMUM = "last_location_of_minimum"
    LEMPEL_ZIV_COMPLEXITY = "lempel_ziv_complexity"
    LENGTH = "length"
    LINEAR_TREND = "linear_trend"
    LINEAR_TREND_TIMEWISE = "linear_trend_timewise"
    LONGEST_STRIKE_ABOVE_MEAN = "longest_strike_above_mean"
    LONGEST_STRIKE_BELOW_MEAN = "longest_strike_below_mean"
    MAX_LANGEVIN_FIXED_POINT = "max_langevin_fixed_point"
    MAXIMUM = "maximum"
    MEAN = "mean"
    MEAN_ABS_CHANGE = "mean_abs_change"
    MEAN_CHANGE = "mean_change"
    MEAN_SECOND_DERIVATIVE_CENTRAL = "mean_second_derivative_central"
    MEDIAN = "median"
    MINIMUM = "minimum"
    NUMBER_CROSSING_M = "number_crossing_m"
    NUMBER_CWT_PEAKS = "number_cwt_peaks"
    NUMBER_PEAKS = "number_peaks"
    PARTIAL_AUTOCORRELATION = "partial_autocorrelation"
    PERCENTAGE_OF_REOCCURRING_DATAPOINTS_TO_ALL_DATAPOINTS = "percentage_of_reoccurring_datapoints_to_all_datapoints"
    PERCENTAGE_OF_REOCCURRING_VALUES_TO_ALL_VALUES = "percentage_of_reoccurring_values_to_all_values"
    PERMUTATION_ENTROPY = "permutation_entropy"
    QUANTILE = "quantile"
    RANGE_COUNT = "range_count"
    RATIO_BEYOND_R_SIGMA = "ratio_beyond_r_sigma"
    RATIO_VALUE_NUMBER_TO_TIME_SERIES_LENGTH = "ratio_value_number_to_time_series_length"
    ROOT_MEAN_SQUARE = "root_mean_square"
    SAMPLE_ENTROPY = "sample_entropy"
    SKEWNESS = "skewness"
    SPKT_WELCH_DENSITY = "spkt_welch_density"
    STANDARD_DEVIATION = "standard_deviation"
    SUM_OF_REOCCURRING_DATA_POINTS = "sum_of_reoccurring_data_points"
    SUM_OF_REOCCURRING_VALUES = "sum_of_reoccurring_values"
    SUM_VALUES = "sum_values"
    SYMMETRY_LOOKING = "symmetry_looking"
    TIME_REVERSAL_ASYMMETRY_STATISTIC = "time_reversal_asymmetry_statistic"
    VALUE_COUNT = "value_count"
    VARIANCE = "variance"
    VARIANCE_LARGER_THAN_STANDARD_DEVIATION = "variance_larger_than_standard_deviation"
    VARIATION_COEFFICIENT = "variation_coefficient"

class TSFreshWrapper(FixedSizeFeatureExtractor):
    """A wrapper for the tsfresh feature extraction. The supported features are specified in the ml4pdm.transformation.TSFreshFeatureCalculators class as Enum.
    """

    def __init__(self, features_name=None) -> None:
        """Constructs the wrapper by specifying the list of features name as enum to be used and its constructor parameters.

        :param features_name: The tsfresh supported feature names that is used for transformation.
        :type features_name: list of ml4pdm.transformation.TSFreshFeatureCalculators keywords. 
        """
        if features_name is not None:
            self._check_features_name(features_name)
            self.default_fc_parameters = self._get_filtered_features(features_name)            
        else:            
            self.default_fc_parameters = None

        self.features_name = features_name

    def _check_features_name(self, features_name: List[Enum]):
        """Check the each element of features_name list with the all the Enum of ml4pdm.transformation.TSFreshFeatureCalculators class.
        If the element of features_name with any of the Enum then it well raises AttributeError.
        
        :param features_name: The tsfresh supported feature names that is used for transformation.
        :type features_name: list of ml4pdm.transformation.TSFreshFeatureCalculators Enum. 
        :raises AttributeError: If the element of features_name list doesn't match with the ml4pdm.transformation.TSFreshFeatureCalculators keyword as Enum. 
        """
        number_of_features = len(features_name) 

        for i in range(number_of_features):
            if not isinstance(features_name[i], TSFreshFeatureCalculators):
                raise AttributeError

    def _get_filtered_features(self, features_name: List[Enum]) -> ComprehensiveFCParameters:
        """Filtered out the features from ComprehensiveFCParameters dict. So each time you remove a key, that feature can no longer be calculated.

        :param features: The tsfresh supported feature names that is used for transformation.
        :type features: list of ml4pdm.transformation.TSFreshFeatureCalculators Enum.
        :return: Filtered features name which is later used for feature extraction.
        :rtype: tsfresh.feature_extraction.settings.ComprehensiveFCParameters
        """
        default_fc_parameters = ComprehensiveFCParameters()
        
        copy_of_fc_parameters = dict(default_fc_parameters)   
        
        no_of_features = len(features_name)

        for key in copy_of_fc_parameters:
            count = 1
            for key_name in features_name:
                if key == key_name.value:                    
                    break
                elif count == no_of_features:
                    del default_fc_parameters[key]
                else:
                    count += 1
        
        return default_fc_parameters
    
    def _pandas_dataframe_wrapper(self, dataset:Dataset) -> DataFrame:
        """This method convert the input datasets to pandas.DataFrame format which is required input format for tsfresh library.

        :param datasets: Datasets that are used for evaluation of pipelines.
        :type datasets: list of Dataset
        :return: Formated input datasets in Stacked DataFrame format.  
        :rtype: pandas.DataFrame
        """
        data = list()       
                
        self._get_instances(dataset=dataset, data= data)
        
        return pd.DataFrame(data, columns=['instance_id', 'timestep', 'sensor','value']).astype({'instance_id':'int32','timestep':'int32','sensor':'string', 'value':'float32'})
    
    def _get_instances(self, dataset:Dataset, data:list):
        """Get the attributes value for each time instance.       
        
        :param datasets: Datasets that are used for evaluation of pipelines.
        :type datasets: list of Dataset
        :param data: Empty list which will be filled for each datasets after converting to instance_id, timestep, sensor, and value format.
        :type data: list       
        """        
        for instance_id, instance in enumerate(dataset.data):
            self._get_attributes(instance_id, instance, dataset, data)

    def _get_attributes(self, instance_id:int, instance:list, dataset:Dataset, data:list):
        """Return the data in instance        
        
        :param instance_id: Instance id at data recorded.
        :type instance_id: int
        :param instance: Data recorded at different time instances.
        :type instance: list
        :param datasets: Datasets that are used for evaluation of pipelines.
        :type datasets: list of Dataset
        :param data: Empty list which will be filled for each datasets after converting to instance_id, timestep, sensor, and value format.
        :type data: list
        """
        for attribute_id, attribute_value in enumerate(instance):
            if isinstance(attribute_value, list):
                number_of_attribute = len(attribute_value) 
                for j in range(number_of_attribute):
                    timestep, value = attribute_value[j]
                    row = [instance_id, timestep, str(dataset.features[attribute_id][0]), value]
                    data.append(row)                    

    def fit(self, x, y=None):
        return self

    def transform(self, x: Dataset) -> Dataset:
        """This method transform the extracted features from tsfresh to Dataset object.

        :param x: Dataset to be transformed.
        :type x: Dataset
        :return: Transformed Dataset object.
        :rtype: Dataset
        """        
        X = self._pandas_dataframe_wrapper(x)        

        extracted_features = extract_features(X, default_fc_parameters=self.default_fc_parameters, kind_to_fc_parameters = None, column_id='instance_id',
                            column_sort='timestep', 
                            column_kind='sensor',
                            column_value='value',
                            chunksize=tsfresh.defaults.CHUNKSIZE,
                            n_jobs=0,
                            show_warnings=tsfresh.defaults.SHOW_WARNINGS,
                            disable_progressbar=True,
                            impute_function=tsfresh.defaults.IMPUTE_FUNCTION,
                            profile=tsfresh.defaults.PROFILING,
                            profiling_filename=tsfresh.defaults.PROFILING_FILENAME,
                            profiling_sorting=tsfresh.defaults.PROFILING_SORTING)        
        
        dataset_transformed = Dataset()
        number_of_features = len(extracted_features.columns)        
        dataset_transformed.features = [(extracted_features.columns[i], NUMERIC()) for i in range(number_of_features)]       
        dataset_transformed.data = np.nan_to_num(np.float32(extracted_features.to_numpy()))                       
        
        return dataset_transformed
