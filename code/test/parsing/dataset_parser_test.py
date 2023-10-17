"""Test script for dataset_parser
"""
import os

from ml4pdm.parsing import DatasetParser

READ_FROM = 'ml4pdm/data/cmapss/FD001_train.pdmff'
WRITE_TO = 'FD001_new.pdmff'
READ_FROM2 = 'test/data/cmapss/sov_example.pdmff'
WRITE_TO2 = 'FD002_new.pdmff'
WRITE_TO3 = 'new_test.pdmff'


def test_dataset_parser():
    """To test dataset parser functionality implemented under parse_from_file() and save_to_file() methods
    """
    dataset_obj = DatasetParser().parse_from_file(READ_FROM)
    DatasetParser().save_to_file(dataset_obj, WRITE_TO)
    os.remove(WRITE_TO)
    # test to check if all the instances are parsed
    assert len(dataset_obj.data) == 100
    # test to check the lengths of data instances parsed
    assert len(dataset_obj.data[0]) == 25

    data_obj = DatasetParser().parse_from_file(READ_FROM2)
    DatasetParser().save_to_file(data_obj, 'sov_new.pdmff')
    os.remove('sov_new.pdmff')
    # test to check if all the instances are parsed
    assert len(data_obj.data) == 3
    # test to check the lengths of data instances parsed
    assert len(data_obj.features) == 3


def test_cmapss_data2():
    """To test dataset parser functionality implemented under get_cmapss_data() method
    """
    # Test = False case, Fetches only Train data from file 'FD002' of cmapss dataset
    train2 = DatasetParser.get_cmapss_data(file='fd002', test=False)
    DatasetParser().save_to_file(train2, WRITE_TO2)
    os.remove(WRITE_TO2)
    # test to check if all the instances are parsed
    assert len(train2.data) == 260
    # test to check the length of the first timeseries attribute of 260th instance parsed
    assert len(train2.data[259][1]) == 316


def test_cmapss_data1():
    """To test dataset parser functionality implemented under get_cmapss_data() method
    """
    train1, test1 = DatasetParser.get_cmapss_data(file='fd004', test=True)
    DatasetParser().save_to_file(test1, WRITE_TO3)
    # test to check if all the instances are parsed
    assert len(train1.data) == 249
    # test to check the lengths of data instances parsed
    assert len(train1.data[0]) == 25
    # test to check if all the target instances are parsed
    assert len(test1.target) == 248
    # test to check if the first target instance is parsed correctly
    assert test1.target[0] == 22

    test1_new = DatasetParser().parse_from_file(WRITE_TO3)
    os.remove(WRITE_TO3)
    # test to check if all the target instances are saved to file
    assert test1_new.relation == 'test_fd004'
    # test to check if the first target instance is saved correctly
    assert test1_new.target[0] == 22
    # test to check if target has instances saved
    assert len(test1_new.target) == 248
