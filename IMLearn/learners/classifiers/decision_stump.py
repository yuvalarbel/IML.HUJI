from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_threshold, best_j, best_sign, best_err = None, None, None, np.inf
        for j in range(X.shape[1]):
            for sign in (-1, 1):
                thr, err = self._find_threshold(X[:, j], y, sign)
                if err < best_err:
                    best_threshold, best_j, best_sign, best_err = thr, j, sign, err
        self.threshold_, self.j_, self.sign_ = best_threshold, best_j, best_sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassification error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sort_index = np.argsort(values)
        sorted_values, sorted_labels = values[sort_index], labels[sort_index]
        thresholds = np.concatenate([[-np.inf], sorted_values[1:], [np.inf]])
        minimum_loss = self._loss_helper(np.full(labels.size, -sign), sorted_labels)
        accumulated_sums = np.cumsum(sorted_labels * sign)
        threshold_errors = np.concatenate([[minimum_loss], minimum_loss - accumulated_sums])
        minimum_error_index = np.argmin(threshold_errors)
        return thresholds[minimum_error_index], threshold_errors[minimum_error_index]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification loss function
        """
        return self._loss_helper(self.predict(X), y)

    def _loss_helper(self, pred: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under weighted misclassification loss function

        Parameters
        ----------
        pred : ndarray of shape (n_samples, )
            Predicted labels

        y : ndarray of shape (n_samples, )
            True labels

        Returns
        -------
        loss : float
            Performance under weighted misclassification loss function
        """
        return sum(np.abs(y[pred != np.sign(y)]))  # / sum(np.abs(y))  # the divisor here should be 1 for our use case
