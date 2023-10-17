from sklearn.pipeline import make_pipeline

from ml4pdm.data import Dataset
from ml4pdm.prediction import HealthIndexEstimator, HICurveEstimator, RemainingUsefulLifetimeEstimator, RULEstimator, WindowedPredictor
from ml4pdm.transformation import FixedSizeFeatureExtractor, RNNAutoencoder, TimeSeriesTransformer, WindowingApproach


class EmbedRUL(RemainingUsefulLifetimeEstimator):
    """This pipeline element can be used to represent the full Embed RUL approach that was described in the following paper:
    Narendhar Gugulothu, Vishnu TV, Pankaj Malhotra, et al. „Predicting Remaining
    Useful Life using Time Series Embeddings based on Recurrent Neural Networks“.
    In: CoRR abs/1709.01073 (2017). arXiv: 1709.01073
    """

    def __init__(self, windowing_approach: TimeSeriesTransformer, rnn_ed: FixedSizeFeatureExtractor,
                 hi_curve_estimator: HealthIndexEstimator, rul_estimator: RemainingUsefulLifetimeEstimator) -> None:
        """Initializes the Embed RUL approach with custom pipeline steps for the different purposes.

        :param windowing_approach: Windowing approach that divides the timeseries into smaller windows
        :type windowing_approach: TimeSeriesTransformer
        :param rnn_ed: The autoencoder that transforms a timeseries into a smaller fixed size feature vector representation
        :type rnn_ed: FixedSizeFeatureExtractor
        :param hi_curve_estimator: The estimator the constructs a health index curve by comparing the embeddings to embedding under normal operation
        :type hi_curve_estimator: HealthIndexEstimator
        :param rul_estimator: The final step of the pipeline that uses run-to-failure HI curves to compare the test instances to and
                              return a RUL prediction
        :type rul_estimator: RemainingUsefulLifetimeEstimator
        """
        super().__init__()
        self.pipeline = make_pipeline(WindowedPredictor(windowing_approach, [rnn_ed], hi_curve_estimator), rul_estimator)

    def fit(self, data: Dataset, label=None, **kwargs) -> "EmbedRUL":
        """Fits the Embed RUL approach by fitting all the different pipeline elements.
        See the above mentioned paper for more details.

        :param data: Dataset that is used to train the Embed RUL approach
        :type data: Dataset
        :return: Self
        :rtype: EmbedRUL
        """
        return self.pipeline.fit(data, label, **kwargs)

    def predict(self, data: Dataset, **kwargs):
        """Predicts the remaining useful lifetime for the machines in the dataset using the Embed RUL approach.
        See the above mentioned paper for more details.

        :param data: Dataset that the RUL predictions should be made for using Embed RUL approach
        :type data: Dataset
        :return: Self
        :rtype: EmbedRUL
        """
        return self.pipeline.predict(data, **kwargs)

    @staticmethod
    def from_params(num_features: int, window_size=30, units=270, learning_rate=0.00584, dropout=0.03263, epochs=50,
                    batch_size=64, verbose=0, plot=False, lambda_param=21.83736, max_time_lag=10, alpha=0.428995, max_rul=120) -> "EmbedRUL":
        """Initializes the Embed RUL approach with the pipeline steps that are used in the above mentioned paper. The default parameters
        were found by hyperparameter optimization and lead to good results.

        :param num_features: Count of feature values for a single timestem (e.g. count of sensors)
        :type num_features: int
        :param window_size: Window size that is used to transform the Dataset, defaults to 30
        :type window_size: int, optional
        :param units: Size of embedding vector and number of GRU units, defaults to 270
        :type units: int, optional
        :param learning_rate: Learning rate used by the Adam optimizer, defaults to 0.00584
        :type learning_rate: float, optional
        :param dropout: Dropout used in the GRU, defaults to 0.03263
        :type dropout: float, optional
        :param epochs: Number of epochs that the fit will train the models for, defaults to 50
        :type epochs: int, optional
        :param batch_size: Batch size used when training the model, defaults to 64
        :type batch_size: int, optional
        :param verbose: Verbosity of the keras training, defaults to 0
        :type verbose: int, optional
        :param plot: Defines whether there should be additional plots that show the training progress and the reconstructed timeseries in
                     comparison to the original one, defaults to False
        :type plot: bool, optional
        :param lambda_param: Defines the factor by which the HI curve difference is divided and therefore controls
                             the notion of similarity, defaults to 21.83736
        :type lambda_param: float, optional
        :param max_time_lag: The maximum amount that the HI curve can be moved while trying to find a good match, defaults to 10
        :type max_time_lag: int, optional
        :param alpha: Defines the percentage of the maximum similarity that is used as a threshold. Only HI curves with a greater or equal
                      similarity will be considered for RUL computation, defaults to 0.428995
        :type alpha: float, optional
        :param max_rul: Limits the maximum RUL prediction to prevent predictions that exceed the total lifetime of a machine, defaults to 120
        :type max_rul: int, optional
        :return: Initialized Embed RUL approach with the default pipeline elements
        :rtype: EmbedRUL
        """
        return EmbedRUL(WindowingApproach(window_size),
                        RNNAutoencoder(num_features, units, learning_rate, window_size, dropout, epochs, batch_size, verbose, plot),
                        HICurveEstimator(window_size), RULEstimator(window_size, lambda_param, max_time_lag, alpha, max_rul))

    @staticmethod
    def from_params_pretrained_rnn(rnn_ed: RNNAutoencoder, lambda_param=21.83736, max_time_lag=10, alpha=0.428995, max_rul=120) -> "EmbedRUL":
        """Initializes the Embed RUL approach with the pipeline steps that are used in the above mentioned paper. This function is used when
        a pretrained RNN autoencoder already exists and it should be used instead of training a new one. The other default parameters were
        found by hyperparameter optimization and lead to good results.

        :param rnn_ed: A pretrained RNN autoencoder that should be used in the instance of the Embed RUL approach
        :type rnn_ed: RNNAutoencoder
        :param lambda_param: Defines the factor by which the HI curve difference is divided and therefore controls
                             the notion of similarity, defaults to 21.83736
        :type lambda_param: float, optional
        :param max_time_lag: The maximum amount that the HI curve can be moved while trying to find a good match, defaults to 10
        :type max_time_lag: int, optional
        :param alpha: Defines the percentage of the maximum similarity that is used as a threshold. Only HI curves with a greater or equal
                      similarity will be considered for RUL computation, defaults to 0.428995
        :type alpha: float, optional
        :param max_rul: Limits the maximum RUL prediction to prevent predictions that exceed the total lifetime of a machine, defaults to 120
        :type max_rul: int, optional
        :return: Initialized Embed RUL approach with the default pipeline elements
        :rtype: EmbedRUL
        """
        return EmbedRUL(WindowingApproach(rnn_ed.window_size), rnn_ed,
                        HICurveEstimator(rnn_ed.window_size), RULEstimator(rnn_ed.window_size, lambda_param, max_time_lag, alpha, max_rul))
