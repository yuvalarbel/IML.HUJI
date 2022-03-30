from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from challenge.preprocessing import Preprocess
from IMLearn.utils import split_train_test
from IMLearn.base import BaseEstimator

import datetime
import numpy as np
import pandas as pd


TRAIN_DATASET = "../datasets/agoda_cancellation_train.csv"
TEST_DATASET = "../datasets/challenge_weeks/test_set_week_1.csv"


def load_data(filename: str, has_tags: bool):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename)
    preprocesser = Preprocess(full_data)

    if not has_tags:
        return preprocesser.run_final()

    features, labels = preprocesser.run()
    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


def print_accuracy_scores(predicted_values: np.ndarray, actual_values: np.ndarray):
    # print error rate, accuracy, precision, recall, f1-score
    print("Error rate / Misclassification Error:",
          np.mean(predicted_values != actual_values))
    print("Accuracy:", np.mean(predicted_values == actual_values))
    # not_predicted = np.logical_not(predicted_values)
    # not_actual = np.logical_not(actual_values)
    # tp = np.sum(np.logical_and(predicted_values, actual_values))
    # fp = np.sum(np.logical_and(predicted_values, not_actual))
    # tn = np.sum(np.logical_and(not_predicted, not_actual))
    # fn = np.sum(np.logical_and(not_predicted, actual_values))
    # print("Precision:", tp / (tp + fp))
    # print("Recall:", tp / (tp + fn))
    # print("Specificity:", tn / (tn + fp))
    # print("False Positive Rate", fp / (fp + tn))
    # print("F1-score:", 2 * tp / (2 * tp + fp + fn))


if __name__ == '__main__':
    final = True

    # Load data and time it
    start_time = datetime.datetime.now()
    df, cancellation_labels = load_data(TRAIN_DATASET, True)
    print("Data preprocessed in:", datetime.datetime.now() - start_time)

    if not final:
        train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
    else:
        train_X, train_y = df, cancellation_labels
        test_X = load_data(TEST_DATASET, False)
        test_y = None

    # Fit model over data and time it
    start_time = datetime.datetime.now()
    estimator = AgodaCancellationEstimator(final).fit(train_X, train_y)
    print("Models fitted in:", datetime.datetime.now() - start_time)

    # Evaluate model on test set
    predictions = estimator.predict(test_X)
    if not final:
        print_accuracy_scores(np.logical_not(np.isnat(predictions)), test_y['real_cancellation_datetime'].notna())
        print(estimator.loss(test_X, test_y))

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "205732621_208785923_314971375.csv")
