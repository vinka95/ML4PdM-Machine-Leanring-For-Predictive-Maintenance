
import os
from functools import partial
from test.parsing.evaluator_config_parser_test import PIPELINE_CONFIGURATION_FILENAME

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline


from ml4pdm.data import EvaluatorConfig

DATASET_FILENAME = os.path.join('test','data','cmapss','FD001_test.pdmff')


def test_evaluator_configuration():
    evaluator_config = init_config()
    assert evaluator_config.dataset_paths is not None and len(evaluator_config.dataset_paths) > 0
    assert evaluator_config.dataset_splitter is not None
    assert evaluator_config.pipeline_paths is not None and len(evaluator_config.pipeline_paths) > 0
    assert evaluator_config.pipelines is not None and len(evaluator_config.pipelines) > 0
    assert evaluator_config.evaluation_metrics is not None and len(evaluator_config.evaluation_metrics) > 0


def init_config():
    list_of_dataset_paths, list_of_pipeline_paths, pipelines, list_of_metrics = [], [], [], []

    list_of_dataset_paths.append(DATASET_FILENAME)

    list_of_pipeline_paths.append(PIPELINE_CONFIGURATION_FILENAME)

    pipelines.append(make_pipeline(NearestNeighbors(n_neighbors=2, radius=0.4)))

    list_of_metrics.append(partial(f1_score, average='macro'))

    return EvaluatorConfig(dataset_paths=list_of_dataset_paths,
                           dataset_splitter=StratifiedKFold(n_splits=3, shuffle=True),
                           pipeline_paths=list_of_pipeline_paths,
                           pipelines=pipelines,
                           evaluation_metrics=list_of_metrics)
