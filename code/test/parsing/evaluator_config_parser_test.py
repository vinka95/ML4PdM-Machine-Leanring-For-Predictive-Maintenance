import os
from test.data import evaluator_config_test

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml4pdm.parsing import EvaluatorConfigParser, PipelineConfigParser

PIPELINE_CONFIGURATION_FILENAME = os.path.join('test', 'parsing', 'PipelineConfiguration.json') 
EVALUATOR_CONFIGURATION_FILENAME = os.path.join('test', 'parsing', 'EvaluatorConfiguration.json')

def test_save_and_parse():
    pipeline = make_pipeline(StandardScaler(with_std=False), GaussianNB(priors=None, var_smoothing=13.37))
    PipelineConfigParser.save_to_file(pipeline, PIPELINE_CONFIGURATION_FILENAME)

    evaluator_config = evaluator_config_test.init_config()

    EvaluatorConfigParser.save_to_file(evaluator_config, EVALUATOR_CONFIGURATION_FILENAME)

    configuration = EvaluatorConfigParser.parse_from_file(EVALUATOR_CONFIGURATION_FILENAME) 

    assert str(evaluator_config.dataset_splitter) == str(configuration._dataset_splitter)

    assert configuration._datasets[0].features is not None and len(configuration._datasets[0].features) > 0
    assert configuration._dataset_splitter is not None
    assert configuration._pipelines[0] is not None and len(configuration._pipelines[0]) > 0
    assert configuration._evaluation_metrics[0] is not None and len(configuration._evaluation_metrics) > 0
    
    os.remove(EVALUATOR_CONFIGURATION_FILENAME)
