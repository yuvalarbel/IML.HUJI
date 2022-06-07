from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_scores = []
    val_scores = []

    m = X.shape[0]
    shuffled_indices = np.array(list(range(m)))
    np.random.shuffle(shuffled_indices)

    folds = np.array_split(shuffled_indices, cv)

    for i in range(cv):
        train_indices = np.concatenate([folds[j] for j in range(cv) if j != i])
        val_indices = folds[i]
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        estimator.fit(X_train, y_train)
        train_scores.append(scoring(y_train, estimator.predict(X_train)))
        val_scores.append(scoring(y_val, estimator.predict(X_val)))

    return float(np.mean(train_scores)), float(np.mean(val_scores))
