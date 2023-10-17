from pandas.core.frame import DataFrame
from ml4pdm.transformation import TimeSeriesTransformer, AttributeFilter
from ml4pdm.data import Dataset, NUMERIC
import pandas as pd
import numpy as np

class MovingWeightedAverage(TimeSeriesTransformer):
    """Calculates one moving weighted value of a time series data.
    """
    
    def __init__(self, span:float, min_periods=0) -> None:
        """Constructs the Moving Weighted Average class constructor using its parameters. 

        :param span: Specify decay in terms of span.
        :type span: float
        :param min_periods: Minimum number of observations in window required to have a value (otherwise result is NA), defaults to 0.
        :type min_periods: int, optional
        """
        self.span = span
        self.min_periods = min_periods
    
    def _convert_to_pandas_dataframe(self, dataset:Dataset)-> DataFrame:
        """[summary]

        :param dataset: Dataset object to be converted to pandas DataFrame.
        :type dataset: Dataset
        :return: Return the pandas.DataFrame after dataset object converted to DataFrame.
        :rtype: DataFrame
        """
        data = list()
        for instance_id, instance in enumerate(dataset.data):
            for attribute_id, attribute_value in enumerate(instance):
                if attribute_id == 1 and isinstance(attribute_value, list):
                    number_of_pair = len(attribute_value)
                    for j in range(number_of_pair):
                        timestep, value = attribute_value[j]
                        row = [instance_id, timestep, value]
                        data.append(row)
                    break
        
        return pd.DataFrame(data, columns=['instance_id', 'timestep', dataset.features[0][0]]).astype({'instance_id':'int32','timestep':'int32', dataset.features[0][0]:'float32'})

    def fit(self, x, y=None):
        return self 

    def transform(self, x: Dataset) -> Dataset:
        """Calculate the moving weighted average of the dataset attribute.

        :param x: Dataset to be transformed.
        :type x: Dataset
        :return: Transformed Dataset object.
        :rtype: Dataset
        """
        dataset_transformed = Dataset()       
        
        filter_obj = AttributeFilter(min_unique_values=8)        

        filter_obj.fit(x)
        
        filtered_dataset = filter_obj.transform(x)        

        data_frame = self._convert_to_pandas_dataframe(filtered_dataset)                        
        
        data_frame_copy = data_frame.copy()

        data_frame = data_frame.drop(columns=[data_frame_copy.columns.values[-1]])        

        ewma_result = data_frame_copy[data_frame_copy.columns.values[-1]].ewm(span=self.span, min_periods=self.min_periods, adjust=False).mean()        

        data_frame[data_frame_copy.columns.values[-1] + '_ewma'] = ewma_result          
        
        number_of_features = len(data_frame.columns)    

        dataset_transformed.data = np.nan_to_num(np.float32(data_frame.to_numpy()))  

        dataset_transformed.features = [(data_frame.columns[i], NUMERIC()) for i in range(number_of_features)]       
        
        return dataset_transformed
    