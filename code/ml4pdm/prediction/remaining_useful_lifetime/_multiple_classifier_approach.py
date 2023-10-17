from typing import Dict, Iterable, List, Tuple, Type

from sklearn.base import ClassifierMixin

from ml4pdm.data import NUMERIC, TIMESERIES, Dataset
from ml4pdm.prediction import RemainingUsefulLifetimeEstimator


class MultipleClassifierApproach(RemainingUsefulLifetimeEstimator):
    """This approach trains multiple classifiers with different failure horizons. This means that a classifier with failure
    horizon 5 will be trained to predict a breakdown that is 5 timesteps away. The approach will evaluate every classifiers
    prediction and conclude a single remaining useful lifetime estimation.

    References:

    * Gian Antonio Susto, Andrea Schirru, Simone Pampuri, Seán F. McLoone, and
      Alessandro Beghi. „Machine Learning for Predictive Maintenance: A Multiple
      Classifier Approach“. In: IEEE Trans. Ind. Informatics 11.3 (2015)
    """

    def __init__(self, failure_horizons: Iterable[int], classifier_class: Type[ClassifierMixin], **classifier_arguments: Dict) -> None:
        """Initializes the multiple classifier approach by instantiating all the needed classifiers.

        :param failure_horizons: List of integers containing the failure horizons. One classifier is created and trained per failure horizon.
        :type failure_horizons: Iterable[int]
        :param classifier_class: Class of classifier to be used
        :type classifier_class: Type[ClassifierMixin]
        """
        super().__init__()
        self.failure_horizons = sorted(failure_horizons, reverse=True)
        self.classifier_class = classifier_class
        self.classifiers = []
        for _ in enumerate(failure_horizons):
            self.classifiers.append(classifier_class(**classifier_arguments))

    def _get_data_and_labels(self, dataset: Dataset, failure_horizon: int) -> Tuple[List[List[float]], List[int]]:
        """Extracts the timeseries data from a Dataset and produces labels for the specified failure horizon.
        For example, with a failure horizon of 5, the last 5 timesteps of every instance are labelled with 1.0. All other timesteps are labelled 0.0.

        :param dataset: Dataset to extract the timeseries from
        :type dataset: Dataset
        :param failure_horizon: Failure horizon that is used to produce the labels
        :type failure_horizon: int
        :return: Returns tuple containing multivariate timeseries and the corresponding labels
        :rtype: Tuple[List[List[float]], List[int]]
        """
        data = []
        labels = []
        _, features = dataset.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))

        for instance in features:
            instance_length = len(instance)
            if instance_length < failure_horizon:
                continue

            labels.extend((instance_length-failure_horizon)*[0]+failure_horizon*[1])
            data.extend(instance)

        return data, labels

    def _fit_single_classifier(self, index: int, dataset: Dataset) -> None:
        """Fits one classifier.

        :param index: Index of the classifier to fit
        :type index: int
        :param dataset: Dataset to fit the classifier on
        :type dataset: Dataset
        """
        data, labels = self._get_data_and_labels(dataset, self.failure_horizons[index])
        self.classifiers[index].fit(data, labels)

    def fit(self, data: Dataset, label=None, **kwargs) -> "MultipleClassifierApproach":
        """Fits all classifiers using different failure horizons that were specified in the constructor.

        :return: Fitted MultipleClassifierApproach
        :rtype: MultipleClassifierApproach
        """
        for i, _ in enumerate(self.failure_horizons):
            self._fit_single_classifier(i, data)
        return self

    def predict(self, data: Dataset, **kwargs) -> List[int]:
        """Iterates over the predictions from every classifier starting with the classifier with the largest failure horizon.
        If a classifier classifies all timesteps as label 0, this means that the failure is at least 'failure horizon' timesteps away.
        This failure horizon is returned as the RUL estimation.

        :param data: Dataset that should be used to predict the RUL
        :type data: Dataset
        :return: List of RUL predictions
        :rtype: List[int]
        """
        _, data = data.get_multivariate_of_type(TIMESERIES(NUMERIC(), NUMERIC()))

        predictions = []
        for instance in data:
            found = False
            for i, classifier in enumerate(self.classifiers):
                classifier_predictions = classifier.predict(instance)
                if classifier_predictions.sum() == 0:
                    predictions.append(self.failure_horizons[i])
                    found = True
                    break

            if not found:
                predictions.append(self.failure_horizons[-1])

        return predictions
