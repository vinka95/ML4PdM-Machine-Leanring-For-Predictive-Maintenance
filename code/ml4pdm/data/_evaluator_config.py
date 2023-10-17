from typing import List

from sklearn.pipeline import Pipeline
class EvaluatorConfig:
    """This class contains the configuration for an Evaluator. It is used when parsing a configuration file.
    """

    def __init__(self, dataset_paths: List[str], dataset_splitter, pipeline_paths: List[str], pipelines: List[Pipeline], evaluation_metrics) -> None:
        """Initializes an Evaluator Config object and stores the below mentioned parameters.

        :param dataset_paths: Dataset file name that contains dataset which are use for evaluation of pipelines.
        :type dataset_paths: list of dataset file name and it's paths.
        :param dataset_splitter: Splitter that is used for splitting the datasets into train and test subsets.
        :type dataset_splitter: object of splitter class.
        :param pipeline_paths: The pipeline configuration file name that contains pipelines which are use for evaluation.
        :type pipeline_paths: list of pipeline configuration file name and it's paths.
        :param pipelines: The pipelines that are evaluated.
        :type pipelines: list of sklearn.pipeline.Pipeline
        :param evaluation_metrics: The metrics that are used to compare the predictions of the pipelines with the true values in the dataset.
        :type evaluation_metrics: list of methods from sklearn.metrics.*
        """
        self.dataset_paths = dataset_paths
        self.dataset_splitter = dataset_splitter
        self.pipeline_paths = pipeline_paths
        self.pipelines = pipelines
        self.evaluation_metrics = evaluation_metrics
