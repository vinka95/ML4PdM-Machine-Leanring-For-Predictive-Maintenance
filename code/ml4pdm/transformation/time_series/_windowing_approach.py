from copy import copy
from typing import List

from ml4pdm.data import ANY, NUMERIC, TIMESERIES, Dataset
from ml4pdm.transformation import TimeSeriesTransformer


class WindowingApproach(TimeSeriesTransformer):
    """This class can be used as a sklearn pipeline element that transforms the timeseries features in a dataset to multiple windows of fixed length.
    """

    def __init__(self, window_size: int, offset: int = 1, rul_target_calculate=True) -> None:
        """Initializes the Windowing Approach.

        :param window_size: Size of the windows that are created.
        :type window_size: int
        :param offset: Offset between two consecutive windows, defaults to 1 = maximum overlap
        :type offset: int, optional
        """
        super().__init__()
        self.window_size = window_size
        self.offset = offset
        self.rul_target_calculate = rul_target_calculate

    def _windowing(self, array: List) -> List[List]:
        """Transforms an array into windows using the predefined settings.

        :param array: Array that will be transformed.
        :type array: List
        :return: Array that is divided into multiple windows
        :rtype: List
        """
        if self.window_size >= len(array):
            return [array]
        return [array[i:i+self.window_size] for i in range(0, len(array)-self.window_size+1, self.offset)]

    def fit(self, data: Dataset = None, label=None, **kwargs) -> "WindowingApproach":
        """Fit does nothing for the Windowing approach.

        :return: Self
        :rtype: WindowingApproach
        """
        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        """Transforms an entire dataset by selecting the timeseries features and applying the windowing function to these.
        All features of other types are copied for each window.

        :param data: Dataset that is to be transformed.
        :type data: Dataset
        :return: Transformed Dataset with instances for every window.
        :rtype: Dataset
        """
        all_features = data.get_features_of_type(ANY())
        features, feature_data = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()), True)

        windowed_dataset = copy(data)
        if data.target is None or len(data.target) == 0:
            data.target = [0 for _ in feature_data]
        transformed_data = []
        transformed_target = []
        for idx, instance in enumerate(feature_data):
            for wini, windowed_instance in enumerate(self._windowing(instance)):
                new_instance = len(all_features)*[None]
                for feature in all_features:
                    if feature in features:
                        fidx = features.index(feature)
                        new_instance[feature[0]] = [inst[fidx] for inst in windowed_instance]
                    else:
                        new_instance[feature[0]] = data.data[idx][feature[0]]
                transformed_data.append(new_instance)
                transformed_target.append(data.target[idx] + len(instance) - self.window_size - self.offset * wini)

        windowed_dataset.data = transformed_data
        if self.rul_target_calculate:
            windowed_dataset.target = transformed_target
        return windowed_dataset
