from copy import deepcopy
from typing import Any, Callable, List, Union

import numpy as np
from sklearn.pipeline import make_pipeline

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.prediction import Predictor
from ml4pdm.transformation import Transformer


class WindowedPredictor(Predictor, Transformer):
    """This class is used to extract data from a dataset, perform some transformations on it and then
    pass it to an aggregator together with the extracted data from the beginning.
    """
    @staticmethod
    def extract_instance_lengths(data: Dataset) -> List[int]:
        """This method counts the timeseries length for the first timeseries feature of every instance and returns a list of counts.

        :param data: Dataset that should be considered for counting
        :type data: Dataset
        :return: A list of timeseries duration for every instance
        :rtype: List[int]
        """
        features = data.get_features_of_type(TIMESERIES(NUMERIC(), NUMERIC()))
        instance_lengths = np.array([len(instance[features[0][0]]) for instance in data.data])
        return instance_lengths

    def __init__(self, windowing_approach: Transformer, windowed_steps: List[Transformer], aggregator: Union[Transformer, Predictor],
                 extraction_function: Callable[[Dataset], Any] = extract_instance_lengths.__func__) -> None:
        """Initializes the windowed predictor and creates a sklearn pipeline for the windowed steps-

        :param windowing_approach: The windowing approach that is used
        :type windowing_approach: Transformer
        :param windowed_steps: The steps that work on the windowed data before aggregation
        :type windowed_steps: List[Transformer]
        :param aggregator: The pipeline element that performs aggregation of the windowed dataset
        :type aggregator: Union[Transformer, Predictor]
        :param extraction_function: This function is used to extract data that will be passed to the aggregator later,
                                    defaults to extract_instance_lengths.__func__
        :type extraction_function: Callable[[Dataset], Any], optional
        """
        super().__init__()
        self.windowing_approach = windowing_approach
        self.windowed_pipeline = make_pipeline(*windowed_steps)
        self.extraction_function = extraction_function
        self.aggregator = aggregator

    def fit(self, data: Dataset, label=None, **kwargs) -> "WindowedPredictor":
        """Fits the windowing approach, the windowed pipeline and the aggregator.
        Before any other action, some information (based on the extraction_function) is extracted. This info is later passed to the aggregator.

        :return: Self
        :rtype: WindowedPredictor
        """
        data_copy = deepcopy(data)
        extracted_data = self.extraction_function(data_copy)
        intermediate_results = self.windowing_approach.fit_transform(data_copy, label)
        intermediate_results = self.windowed_pipeline.fit_transform(intermediate_results, label)
        self.aggregator.fit(intermediate_results, None, extracted_data=extracted_data)
        return self

    def transform(self, data: Dataset, **kwargs) -> Dataset:
        """Transforms using the windowing approach, the windowed pipeline and the aggregator.
        Before any other action, some information (based on the extraction_function) is extracted. This info is later passed to the aggregator.

        :param data: Dataset to be transformed
        :type data: Dataset
        :return: Transformed dataset
        :rtype: Dataset
        """
        extracted_data = self.extraction_function(data)
        intermediate_results = self.windowing_approach.transform(data)
        intermediate_results = self.windowed_pipeline.transform(intermediate_results)
        return self.aggregator.transform(intermediate_results, extracted_data=extracted_data)

    def predict(self, data: Dataset, **kwargs) -> Any:
        """Predicts using the windowing approach, the windowed pipeline and the aggregator.
        Before any other action, some information (based on the extraction_function) is extracted. This info is later passed to the aggregator.

        :param data: Dataset to be transformed and predicted
        :type data: Dataset
        :return: Predictions from dataset
        :rtype: Dataset
        """
        extracted_data = self.extraction_function(data)
        intermediate_results = self.windowing_approach.transform(data)
        intermediate_results = self.windowed_pipeline.transform(intermediate_results)
        return self.aggregator.predict(intermediate_results, extracted_data=extracted_data)
