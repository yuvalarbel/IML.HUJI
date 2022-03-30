from __future__ import annotations
from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_auc_score

from IMLearn.base import BaseEstimator


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """
    NUM_ESTIMATORS = 500
    LOSS_THRESHOLD = 60 * 60 * 24 * 3
    ORIGINAL_DATES_COLS = ['X_booking_datetime_original', 'X_checkin_date_original']
    Y_COLUMNS = ['time_to_cancel', 'cancel_time_to_checkin', 'real_cancellation_datetime']

    MIN_DATE_THRESHOLD = '2018-12-05'
    MAX_DATE_THRESHOLD = '2018-12-15'

    N_NEIGHBORS = 5

    def __init__(self, final) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.final = final
        self._clf = RandomForestClassifier(n_estimators=self.NUM_ESTIMATORS)
        self._reg_after_booking = KNeighborsRegressor(self.N_NEIGHBORS, weights='distance')
        self._reg_before_checkin = KNeighborsRegressor(self.N_NEIGHBORS, weights='distance')

    @classmethod
    def _get_y(cls, y, col_index):
        return y[cls.Y_COLUMNS[col_index]]

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        raw_X = X[X.columns.difference(self.ORIGINAL_DATES_COLS)]

        y_cancelled_classes = self._get_y(y, 2).notna()
        self._clf.fit(raw_X, y_cancelled_classes)

        reg_X = raw_X[y_cancelled_classes == 1]
        reg_y_after_booking = self._get_y(y, 0)[y_cancelled_classes == 1]
        reg_y_before_checkin = self._get_y(y, 1)[y_cancelled_classes == 1]

        self._reg_after_booking.fit(reg_X, reg_y_after_booking)
        self._reg_before_checkin.fit(reg_X, reg_y_before_checkin)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        raw_X = X[X.columns.difference(self.ORIGINAL_DATES_COLS)]
        classification_prediction = self._clf.predict(raw_X)

        reg_X = raw_X[classification_prediction == 1]
        regression_prediction_after_booking = self._reg_after_booking.predict(reg_X)
        regression_prediction_before_checkin = self._reg_before_checkin.predict(reg_X)

        prediction_a = X.loc[classification_prediction == 1][self.ORIGINAL_DATES_COLS[0]] + pd.to_timedelta(regression_prediction_after_booking, unit='s')
        prediction_b = X.loc[classification_prediction == 1][self.ORIGINAL_DATES_COLS[1]] - pd.to_timedelta(regression_prediction_before_checkin, unit='s')

        cancellation_time = prediction_a + ((prediction_b - prediction_a) / 2)

        results = X[[]]
        if not self.final:
            results.loc[cancellation_time.index, 'prediction'] = cancellation_time
            return results.prediction

        is_in_dates = np.logical_and(cancellation_time >= self.MIN_DATE_THRESHOLD,
                                     cancellation_time <= self.MAX_DATE_THRESHOLD)

        results.loc[is_in_dates.index, 'prediction'] = is_in_dates

        return np.where(classification_prediction == 1, results.prediction, False).astype(int)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        predictions = self.predict(X)
        real_cancel_times = self._get_y(y, 2)

        if self.final:
            return roc_auc_score(real_cancel_times.notna(), predictions)

        cancel_time_predictions = (real_cancel_times - predictions) / np.timedelta64(1, 's') < self.LOSS_THRESHOLD
        correct_or_incorrect = np.where(np.isnat(predictions),
                                        real_cancel_times.isna(),
                                        cancel_time_predictions).astype(int)

        print("Error rate / Misclassification Error:", np.mean(1 - correct_or_incorrect))
        print("Accuracy:", np.mean(correct_or_incorrect))

        return roc_auc_score(correct_or_incorrect, np.ones(len(correct_or_incorrect)))
