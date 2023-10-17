import copy
from statistics import mean
from typing import Dict, List

import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from ml4pdm.data import Dataset
from ml4pdm.prediction import RemainingUsefulLifetimeEstimator
from ml4pdm.transformation import AttributeFilter


class SVREstimator(RemainingUsefulLifetimeEstimator):
    """This wrapper is based on a direct approach for the RUL estimation without the need to define health states based on an SVR-RUL
    model, proposed in the following paper:

    R. Khelif, B. Chebel-Morello, S. Malinowski, E. Laajili, F. Fnaiech and N. Zerhouni, 
    "Direct Remaining Useful Life Estimation Based on Support Vector Regression," 
    in IEEE Transactions on Industrial Electronics, vol. 64, no. 3, pp. 2276-2285, March 2017, doi: 10.1109/TIE.2016.2623260.
    """

    def __init__(self, window_size, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, c_value=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1, cv=10) -> None:
        """This instantiates the SVR model and its parameters that acts as an estimator and used to predict the RUL for given training instances.

        :param window_size: Size of windows the timeseries data going to be split into.
        :type window_size: int
        :param kernel: Specifies the kernel type to be used in the algorithm., defaults to 'rbf'
        :type kernel: str, optional
        :param degree: Degree of the polynomial kernel function, defaults to 3
        :type degree: int, optional
        :param gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’, defaults to 'scale'
        :type gamma: str, optional
        :param coef0: Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’, defaults to 0.0
        :type coef0: float, optional
        :param tol: Tolerance for stopping criterion, defaults to 1e-3
        :type tol: [type], optional
        :param c_value: Regularization parameter, defaults to 1.0
        :type c_value: float, optional
        :param epsilon: Epsilon in the epsilon-SVR model, defaults to 0.1
        :type epsilon: float, optional
        :param shrinking: [Whether to use the shrinking heuristic, defaults to True
        :type shrinking: bool, optional
        :param cache_size: Specify the size of the kernel cache, defaults to 200
        :type cache_size: int, optional
        :param verbose: Enable verbose output, defaults to False
        :type verbose: bool, optional
        :param max_iter: Hard limit on iterations within solver, or -1 for no limit, defaults to -1
        :type max_iter: int, optional
        :param cv: etermines the cross-validation splitting strategy, defaults to 10
        :type cv: int, optional
        """

        super().__init__()
        self.parameters = dict()
        self.C_list, self.epsilon_list, self.gamma_list, self.instance_nums = [], [], [], []

        self.C_list.append(c_value)
        self.epsilon_list.append(epsilon)
        self.gamma_list.append(gamma)

        self.parameters['C'] = self.C_list
        self.parameters['epsilon'] = self.epsilon_list
        self.parameters['gamma'] = self.gamma_list
        self.window_size = window_size

        self.model = svm.SVR(kernel=kernel, degree=degree, gamma=gamma,
                             coef0=coef0, tol=tol, C=c_value, epsilon=epsilon, shrinking=shrinking,
                             cache_size=cache_size, verbose=verbose, max_iter=max_iter)

        self.model = GridSearchCV(self.model, self.parameters, cv=cv, scoring='neg_mean_squared_error', verbose=verbose, n_jobs=-1)

    def fit(self, data: Dataset, label=None,  **kwargs) -> "SVR Estimator Fit":
        """This extends the fit method of SVR according to details in the paper mentioned above. 
        Computes the training labels before fitting model and then returns the fitted model.  

        :param data: Dataset that should be used for training SVR estimator
        :type data: Dataset
        :return: self
        :rtype: SVREstimator
        """
        y = data.target
        length_labels = len(y)
        data_copy = copy.deepcopy(data)
        # Compute Training Labels
        data_df = pd.DataFrame(data_copy.data)
        new_train_labels = []
        for i in range(length_labels):
            id_eng = i + 1
            ins_df = data_df[data_df[0] == id_eng]
            ins_count = ins_df.shape[0]
            start_label = y[i]
            for j in range(ins_count):
                new_train_labels.append(start_label - (j+1)*self.window_size)

        # Remove Engine ID column
        filter_obj = AttributeFilter(remove_indices=[0])
        filter_obj.fit(data)
        filtered_dataset = filter_obj.transform(data)

        return self.model.fit(filtered_dataset.data, new_train_labels)

    def predict(self, data: Dataset,  **kwargs) -> List[int]:
        """This extends the predict method of SVR model according to details in the paper mentioned above. 
        Computes the mean of the predictions of each window to get predictions for individual instance and returns them. 

        :param data: Dataset that the RUL predictions should be made for using SVR estimator
        :type data: Dataset
        :return: Predictions
        :rtype: List[int]
        """
        data_copy = copy.deepcopy(data)
        data_df = pd.DataFrame(data_copy.data)
        self.instance_nums = data_df[0]

        # Remove Engine ID column
        filter_obj = AttributeFilter(remove_indices=[0])
        filter_obj.fit(data)
        filtered_dataset = filter_obj.transform(data)

        predictions = self.model.predict(filtered_dataset.data)

        # Combine Predictions
        pred_tuples = list(zip(self.instance_nums.tolist(), predictions))
        pred_df = pd.DataFrame(pred_tuples)

        new_predictions = []
        for i in self.instance_nums.unique():
            pred_ins_df = pred_df[pred_df[0] == i]
            ins_preds = []
            num_of_remaining_windows = pred_ins_df.shape[0] - 1
            for j in range(pred_ins_df.shape[0]):
                ins_preds.append(pred_ins_df.iloc[j][1] - (self.window_size * num_of_remaining_windows))
                num_of_remaining_windows = num_of_remaining_windows - 1
            new_predictions.append(mean(ins_preds))

        return new_predictions
