from typing import List

from sklearn.pipeline import Pipeline
import jsonpickle

from ml4pdm.data import Dataset, EvaluatorConfig
from ml4pdm.evaluation import Evaluator
from ml4pdm.parsing import DatasetParser, PipelineConfigParser


class EvaluatorConfigParser:
    """Contains method to transform Evaluator configration into JSON and back.
    """
    @staticmethod
    def _get_datasets(paths: list) -> List[Dataset]:
        """Get the respective dataset from the list of dataset paths.

        :param paths: List of dataset file name and paths.
        :type paths: list
        :return: List of dataset in pdmff format.
        :rtype: list<ml4pdm.data.Dataset>
        """
        datasets = []
        no_dataset_paths = len(paths)
        if no_dataset_paths > 0:
            for i in range(no_dataset_paths):
                datasets.append(DatasetParser.parse_from_file(paths[i]))

        return datasets

    @staticmethod
    def _get_pipelines(paths: list, strings: list) -> List[Pipeline]:
        """It will give list of pipeline after fetching respective pipeline configuration from file and pipeline as JSON string.

        :param paths: List of pipeline file name and paths.
        :type paths: list
        :param strings: List of pipelines mentioned as JSON string.
        :type strings: list
        :return: List of pipelines.
        :rtype: list<sklearn.pipeline.Pipeline>
        """
        pipelines = []
        no_pipeline_paths, no_pipeline_strings = len(paths), len(strings)
        if no_pipeline_paths > 0 or no_pipeline_strings > 0:
            if no_pipeline_paths > 0:
                for j in range(no_pipeline_paths):
                    pipelines.append(PipelineConfigParser.parse_from_file(paths[j]))

            if no_pipeline_strings > 0:
                for k in range(no_pipeline_strings):
                    pipelines.append(PipelineConfigParser.parse_from_string(jsonpickle.encode(strings[k])))
        return pipelines

    @staticmethod
    def parse_from_file(path: str) -> Evaluator:
        """Parses a Pre-defined Evaluator Configuration as a JSON file in input and in return it will give Evaluator object.
        Evaluator object can be passed to  evaluate method which will evaluate the model.

        :param path: Evaluator Configuration file name which contains the description in JSON format.
        :type path: string
        :return Evaluator: Object that contains datasets, pipelines, dataset_splitter and evaluation_metrics.
        :rtype: ml4pdm.evaluation.Evaluator
        """
        with open(path, "r") as file:
            configuration = jsonpickle.decode(file.read())

        return Evaluator(datasets=EvaluatorConfigParser._get_datasets(configuration.dataset_paths),
                         pipelines=EvaluatorConfigParser._get_pipelines(configuration.pipeline_paths, configuration.pipelines),
                         dataset_splitter=configuration.dataset_splitter,
                         evaluation_metrics=configuration.evaluation_metrics)

    @ staticmethod
    def save_to_file(evaluator: EvaluatorConfig, path: str) -> None:
        """Saves a configured Evaluator to a JSON file.

        :param evaluator: Evaluator that will be saved.
        :type evaluator: ml4pdm.data.evaluator_config.EvaluatorConfig
        :param path: Path and filename of the JSON file that is created.
        :type path: string
        """
        output = jsonpickle.encode(evaluator, indent=4)
        with open(path, "w") as file:
            file.write(output)
