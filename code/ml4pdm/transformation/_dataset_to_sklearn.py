from typing import Any, Callable, List

from sklearn.pipeline import make_pipeline

from ml4pdm.data import Dataset
from ml4pdm.transformation import Transformer


class DatasetToSklearn(Transformer):
    """The transformer class to extract sklearn-usable format from the Dataset object passed in the ml4pdm library.
    """

    def __init__(self, extractor: Callable[[Dataset], Any] = None):
        """Constructor with extractor function, which is used to extract the data in the correct format.

        :param extractor: Extractor function which is called on Dataset.data as an input, defaults to None
        :type extractor: Callable[[Dataset], Any], optional
        """
        self.extractor = extractor

    def fit(self, data: Dataset, label=None, **kwargs) -> "DatasetToSklearn":
        """fit does nothing but returns this object

        :return: Self
        :rtype: DatasetToSklearn
        """
        return self

    def transform(self, data: Dataset, **kwargs):
        """Extracts the .data from the data object using the extractor function.
        extractor(data.data)
        If the extractor is not defined, this function returns data.data

        :param data: Dataset, from which the .data is extracted using the extractor function
        :type data: Dataset
        :return: extracted data from the dataset object data
        :rtype: array-like
        """
        if self.extractor is None:
            return data.data
        return self.extractor(data)


class ML4PdM(Transformer):
    """Wrapper for ml4pdm classes. This wrapper expects a Dataset object as input and outputs a format depending on the specified extractor.
    """

    def __init__(self, ml4pdm_steps: List[Transformer], extractor: Callable[[Dataset], Any] = None):
        """Constructor for the ML4PdM wrapper.

        :param ml4pdm_steps: The pipeline steps to be executed in this wrapper
        :type ml4pdm_steps: List[Transformer]
        :param extractor: The extractor function. If this is None, the Dataset.data is returned in the transform, defaults to None
        :type extractor: Callable[[Dataset], Any], optional
        """
        self.pipeline = make_pipeline(*ml4pdm_steps)
        self.extractor_element = DatasetToSklearn(extractor)

    def fit(self, data: Dataset, label=None, **kwargs) -> "ML4PdM":
        """Fits the pipeline consisting of the specified ml4pdm pipeline elements.
        :param data: Dataset to be fit on
        :type data: Dataset
        :return: Self
        :rtype: ML4PdM
        """
        self.pipeline.fit(data, data.target)
        return self

    def transform(self, data: Dataset, **kwargs):
        """Transformes the specified dataset using the specified pipeline elements and then extracts a format using the extractor function.

        :param data: The dataset to be transformed and then extracted from
        :type data: Dataset
        :return: Extracted instances
        :rtype: array-like
        """
        return self.extractor_element.transform(self.pipeline.transform(data))
