from copy import deepcopy
from typing import Any, Callable, List, Tuple

from sklearn.base import TransformerMixin

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.transformation import Transformer


class SklearnWrapper(Transformer):
    """This class serves as a wrapper for any sklearn transformer. Typically sklearn transformers only work on arrays.
    This class will convert the Dataset to an array, transform it and convert it back to a Dataset.
    It can be used in a Pipeline prior to any ML4Pdm step.
    """

    @staticmethod
    def extract_timeseries(data: Dataset) -> Tuple[List[List], List[Tuple[int, str]]]:
        """Extracts only the timeseries features from the dataset and converts them to a multivariate layout.
        Should be used with 'rebuild_timeseries'.

        :param data: Dataset to extract the timeseries from
        :type data: Dataset
        :return: Tuple of extracted data and list of features for later reassembly
        :rtype: Tuple[List[List], List[Tuple[int, str]]]
        """
        features, result = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        return result, features

    @staticmethod
    def rebuild_timeseries(old_data: Dataset, data: List[List], features: List[Tuple[int, str]]) -> Dataset:
        """Sets the timeseries features on an existing from a transformed data array and a list of features.
        Should be used with 'extract_timeseries'.

        :param old_data: Dataset that will be modified with the new data
        :type old_data: Dataset
        :param data: Transformed data array that should be inserted into a Dataset again
        :type data: List[List]
        :param features: List of features that were extracted earlier
        :type features: List[Tuple[int, str]]
        :return: Returns old_dataset which was modified with the new timeseries data
        :rtype: Dataset
        """
        old_data.set_from_multivariate(features, data)
        return old_data

    @staticmethod
    def extract_timeseries_concatenated(data: Dataset) -> Tuple[List[List], Tuple[List[Tuple[int, str]], List[int]]]:
        """Extracts only the timeseries features from the dataset and converts them to a multivariate layout.
        This function will also merge all timeseries of the instances for sklearn transformers like MinMaxScaler to deal with it easily.
        Should be used with 'rebuild_timeseries_concatenated'.

        :param data: Dataset to extract the timeseries from
        :type data: Dataset
        :return: Tuple of extracted data and list of features + list of instance lengths for later reassembly
        :rtype: Tuple[List[List], List[Tuple[int, str]]]
        """
        features, result = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        concat_result = []
        lengths = []
        for instance in result:
            lengths.append(len(instance))
            concat_result.extend(instance)
        return concat_result, (features, lengths)

    @staticmethod
    def rebuild_timeseries_concatenated(old_data: Dataset, data: List[List], info: Tuple[List[Tuple[int, str]], List[int]]) -> Dataset:
        """Sets the timeseries features on an existing from a transformed data array and a list of features.
        This function will split the merged timeseries into the different instances again.
        Should be used with 'extract_timeseries_concatenated'.

        :param old_data: Dataset that will be modified with the new data
        :type old_data: Dataset
        :param data: Transformed data array that should be inserted into a Dataset again
        :type data: List[List]
        :param features: List of features that were extracted earlier + List of instance lengths to split the instances again
        :type features: List[Tuple[int, str]]
        :return: Returns old_dataset which was modified with the new timeseries data
        :rtype: Dataset
        """
        split_data = []
        start = 0
        for length in info[1]:
            split_data.append(data[start:start+length])
            start += length

        old_data.set_from_multivariate(info[0], split_data)
        return old_data

    def __init__(self, transformer: TransformerMixin,
                 before: Callable[[Dataset], Tuple[Any, Any]] = extract_timeseries.__func__,
                 after: Callable[[Dataset, Any, Any], Dataset] = rebuild_timeseries.__func__) -> None:
        """Initializes the sklearn wrapper.

        :param transformer: Instance of a single sklearn transformer
        :type transformer: TransformerMixin
        :param before: transformation function from Dataset to array and additional info, defaults to extract_timeseries.__func__
        :type before: Callable[[Dataset], Tuple[Any, Any]], optional
        :param after: transformation function from old dataset, transformed data and additional info to Dataset,
                      defaults to rebuild_timeseries.__func__
        :type after: Callable[[Dataset, Any, Any], Dataset], optional
        """
        super().__init__()
        self.transformer = transformer
        self.before = before
        self.after = after

    def fit(self, data: Dataset, label=None, **kwargs) -> "SklearnWrapper":
        """Fits the underlying transformer by first applying the 'before' function.

        :return: Fitted SklearnWrapper
        :rtype: SklearnWrapper
        """
        modified_data, _ = self.before(data)
        self.transformer.fit(modified_data)
        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        """Transforms the data using the underlying transformer and applying the 'before' and 'after' functions.

        :param data: Data to be transformed
        :type data: Dataset
        :return: Transformed data
        :rtype: Dataset
        """
        data = deepcopy(data)
        modified_data, info = self.before(data)
        modified_data = self.transformer.transform(modified_data)
        modified_data = self.after(data, modified_data, info)
        return modified_data
