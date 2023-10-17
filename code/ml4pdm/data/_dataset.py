"""Contains Dataset class
"""
from copy import deepcopy
from random import random
from typing import List, Tuple

import numpy as np
from ml4pdm.data import ANY, TIMESERIES, AttributeType


class Dataset:
    """Dataset class contains different dataset elements as parameters.
    """

    def __init__(self):
        """Initializes a Dataset object and stores the specified parameters for the dataset.
        """
        self.data = []
        self.target = []
        self.features = []
        self.description = ''
        self.relation = ''
        self.target_name = ''
        self.target_type = ''

    def __str__(self):
        """Return the dataset class attributes in following string format.
        i.e.- "Dataset(Features=[('id', NUMERIC), ('setting1', TIMESERIES(NUMERIC:NUMERIC)),....] , Length of data=100 , Length of Target=100)"

        :return: Return the string which have dataset features and length of data and target.
        :rtype: str
        """
        return 'Dataset(Features=' + str(self.features) + ' , Length of data=' + str(len(self.data)) + ' , Length of Target='+ str(len(self.target)) + ')'

    def get_features_of_type(self, type_name: AttributeType) -> List[Tuple[int, str]]:
        """Gets a list of feature indices and names that match the given type.

        :param type_name: Type of feature that should be filtered for.
        :type type_name: AttributeType
        :return: List of Tuples containing the feature index and name.
        :rtype: List[Tuple[int, str]]
        """
        results = []
        for i, feature in enumerate(self.features):
            if isinstance(type_name, ANY) or repr(feature[1]) == repr(type_name):
                results.append((i, feature[0]))
        return results

    def get_multivariate_of_type(self, type_name: AttributeType, keep_timestamp=False) -> Tuple[List[Tuple[int, str]], List[List]]:
        """Returns feature data that matches the given attribute type in a multivariate layout.
        Example shape: (instance, timestep, sensor, (timestep+)value).

        :param type_name: Type of feature that should be filtered for
        :type type_name: AttributeType
        :param keep_timestamp: Defines whether tuples containing timestep and value are returned or only values are returned, defaults to False
        :type keep_timestamp: bool, optional
        :raises Exception: When no features match the given type, an exception is thrown
        :return: Selected features and multi-dimensional array containing data in a multivariate layout
        :rtype: Tuple[List[Tuple[int, str]], List[List]]
        """
        data = []
        features = self.get_features_of_type(type_name)

        if len(features) == 0:
            raise LookupError("No values found for specified type in this dataset!")

        for instance in self.data:
            features_data = []
            if keep_timestamp:
                features_data = [[data for data in instance[feature[0]]] for feature in features]
            else:
                features_data = [[data[1] for data in instance[feature[0]]] for feature in features]
            zipped_features = [list(item) for item in zip(*features_data)]
            data.append(zipped_features)

        return features, data

    def set_from_multivariate(self, features: List[Tuple[int, str]], data: List[List], add_timestamp=True) -> None:
        """Updates the dataset with modified data that was previously transformed using 'get_multivariate_of_type'.

        :param features: List of features that are to be set
        :type features: List[Tuple[int, str]]
        :param data: Data that the dataset will be updated with
        :type data: List[List]
        :param add_timestamp: Defines whether a timestamp has to be added if it is not contained in the data, defaults to True
        :type add_timestamp: bool, optional
        """
        for idx, _ in enumerate(data):
            for fidx, feature in enumerate(features):
                if add_timestamp:
                    self.data[idx][feature[0]] = [(tidx+1, sensors[fidx]) for tidx, sensors in enumerate(data[idx])]
                else:
                    self.data[idx][feature[0]] = [sensors[fidx] for sensors in data[idx]]

    def get_time_series_data_as_array(self):
        """gets the data with all timeseries attributes transformed into simple lists of values.
        To change the .data properly in this object, you need to set .data_w_list_ts to None.
        Otherwise this method will return the old transformed data

        :return: data with timeseries features transformed into lists.
        :rtype: list of instances
        """
        ret = []
        for instance in self.data:
            new_instance = []

            for idf, feature in enumerate(instance):

                if isinstance(self.features[idf][1], TIMESERIES):
                    new_instance.append(np.asarray([pair[1] for pair in feature]))
                else:
                    new_instance.append(feature)
            ret.append(new_instance)
        return ret

    def generate_simple_cut_dataset(self, cut_repeats: int = 1, min_length: int = 0, max_length: int = None):
        """generates an augmented dataset out of full maintenance data.
        That type of data is run-to-failure which means that the last timestep in the TIMESERIES attributes are
        the last timestep that could be recorded of the sensors before failure occured.
        This method generates a simple cut dataset with labeling for a RUL (Remaining Useful Lifetime) regression task from the dataset.
        You can specify the number of repeated cuts and the minimum and maximum length of the instances.

        :param cut_repeats: number of times each instance gets cut to create an instance in the resulting dataset, defaults to 1
        :type cut_repeats: int, optional
        :param min_length: minimum length of the TIMESERIES features, defaults to 0
        :type min_length: int, optional
        :param max_length: maximum length of the TIMESERIES features, defaults to None
        :type max_length: int, optional
        :return: new dataset with generated instances for a RUL regression task
        :rtype: Dataset
        """
        dataset = deepcopy(self)
        data = dataset.data
        target = []
        cut_data = []
        min_length = min_length - 1
        for instance in data:
            min_time = min([instance[idf][0][0] for idf, feat in enumerate(dataset.features) if isinstance(feat[1], TIMESERIES)])
            max_time = max([instance[idf][-1][0] for idf, feat in enumerate(dataset.features) if isinstance(feat[1], TIMESERIES)])
            for _ in range(cut_repeats):
                original_length = (max_time - min_time)
                cut_max = max_length
                if max_length is None or original_length < max_length:
                    cut_max = original_length
                actual_cut_point = max_time - min_length - (cut_max - min_length) * random()

                target.append(max_time - int(actual_cut_point))
                cut_data.append([[pair for pair in attr if pair[0] < actual_cut_point] if isinstance(
                    dataset.features[idf][1], TIMESERIES) else attr for idf, attr in enumerate(instance)])
        dataset.data = cut_data
        dataset.target = target
        return dataset

class DatasetSummary():
    """It will print the Dataset Summary in string format.
    """

    def fit(self, x, y=None):
        return self
    
    def transform(self, dataset:Dataset)-> "Dataset":
        """The print method will call the __str__() of Dataset class which will return dataset summary in string format.
        
        :param data: Dataset object which will contain data at the transformation.
        :type data: Dataset
        :return: Self
        :rtype: DatasetSummary
        """        
        print(dataset.__str__())        
        return dataset      
                    