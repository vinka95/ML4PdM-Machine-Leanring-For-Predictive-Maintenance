import os
from test.parsing.evaluator_config_parser_test import PIPELINE_CONFIGURATION_FILENAME

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml4pdm.parsing import PipelineConfigParser

def test_save_and_parse():
    pipeline = make_pipeline(StandardScaler(with_std=False), GaussianNB(priors=None, var_smoothing=13.37))
    print("ORIGINAL:", pipeline)
    PipelineConfigParser.save_to_file(pipeline, PIPELINE_CONFIGURATION_FILENAME)

    restored = PipelineConfigParser.parse_from_file(PIPELINE_CONFIGURATION_FILENAME)
    print("RESTORED:", restored)
    
    assert str(pipeline) == str(restored)

    os.remove(PIPELINE_CONFIGURATION_FILENAME)
