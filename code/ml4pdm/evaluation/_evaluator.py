"""This module contains the Evaluator class which is used to evaluate pipelines on datasets using metrics.
"""
from copy import copy

import numpy as np

from ml4pdm.data import Dataset


class Evaluator:
    """This class contains an evaluate method that evaluates pipelines on datasets, that get split via the dataset splitter and then evaluated
    using a list of evaluation metrics. All these are specified in the constructor.
    """

    def __init__(self, datasets, pipelines, dataset_splitter, evaluation_metrics):
        """Constructor parameters of member variables for datasets, pipelines, dataset splitter and evaluation metrics:

        :param datasets: datasets that are used for evaluation of pipelines
        :type datasets: list of Dataset of length n_datasets
        :param pipelines: the pipelines that are evaluated
        :type pipelines: list of sklearn.pipeline.Pipeline length n_pipelines
        :param dataset_splitter: splitter that is used for splitting the datasets into train and test subsets
        :type dataset_splitter: object of splitter class (e.g. sklearn.model_selection.KFold)
        :param evaluation_metrics: the metrics that are used to compare the predictions of the pipelines with the true values in the dataset.
        :type evaluation_metrics: list of function of length n_metrics
        """
        self._datasets = datasets
        self._pipelines = pipelines
        self._dataset_splitter = dataset_splitter
        self._evaluation_metrics = evaluation_metrics
        self.full_y_pred_per_pipeline = None

    def evaluate(self):
        """This method loops first over the list of datasets and splits them using the dataset_splitter into training and test subdatasets.
        Then each pipeline is fitted on the training set and the results for the testing set are predicted.
        The resulting predictions are then compared with the true target values using all the metrics.
        The metric values are then averaged over all the splits of the dataset and returned for each dataset, each pipeline and each metric seperately.

        :return: Averaged evaluation results for all datasets evaluating all pipelines using all metrics specified in the constructor of this Evaluator object.
        :rtype: ndarray of shape (n_datasets, n_pipelines, n_metrics)
        """
        metric_per_pipeline_per_dataset = []

        for dataset in self._datasets:
            metric_per_pipeline_single_dataset = []
            data_x = dataset.data
            target_y = dataset.target

            for train_index, test_index in self._dataset_splitter.split(data_x, target_y):

                # split into train and test subsets using the indices provided by the splitter
                x_train, x_test = data_x[train_index], data_x[test_index]
                y_train, y_test = target_y[train_index], target_y[test_index]

                # build datasets from these splits
                train_dataset = copy(dataset)
                train_dataset.data = x_train
                train_dataset.target = y_train

                test_dataset = copy(dataset)
                test_dataset.data = x_test
                test_dataset.target = y_test

                # evaluate this split and append results
                metric_per_pipeline_single_dataset.append(self.evaluate_train_test_split(train_dataset, test_dataset))

            # aggregate the results using the mean over the different splits:
            metric_per_pipeline_per_dataset.append((np.array(metric_per_pipeline_single_dataset).mean(axis=0).tolist()))
        return np.array(metric_per_pipeline_per_dataset)

    def evaluate_train_test_split(self, train_dataset: Dataset, test_dataset: Dataset):
        """This method evaluates the specified train test split using all pipelines and all metrics.

        :param train_dataset: Dataset for training the pipelines
        :type train_dataset: Dataset
        :param test_dataset: Dataset for testing the pipelines
        :type test_dataset: Dataset
        :return: evaluation results with dimensions (n_pipelines, n_metrics)
        :rtype: list of (list of float)
        """
        self.full_y_pred_per_pipeline = []
        one_split_metric_evaluations = []

        for pidx, pipeline in enumerate(self._pipelines):
            one_split_metric_evaluations.append([])

            # fit and predict on the pipeline
            pipeline.fit(train_dataset, train_dataset.target)
            y_pred = pipeline.predict(test_dataset)
            self.full_y_pred_per_pipeline.append(y_pred)
            # evaluate using all metrics
            for metric in self._evaluation_metrics:
                one_split_metric_evaluations[pidx].append(metric(test_dataset.target, y_pred))

        return one_split_metric_evaluations
