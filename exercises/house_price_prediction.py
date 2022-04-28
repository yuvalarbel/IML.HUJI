from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATE_FORMAT = "%Y%m%dT000000"
ONE_DAY = np.timedelta64(1, 'D')
BEGINNING_DATE = pd.Timestamp('2014-01-01')


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename).dropna()
    data = data[data.price > 0]
    days = (pd.to_datetime(data.date, format=DATE_FORMAT) - BEGINNING_DATE) / ONE_DAY
    data['days_from_2014'] = days
    data = data[np.logical_and(data.bathrooms != 0, data.bedrooms != 0)]
    data['zipcategory'] = data.zipcode.astype('category').cat.codes.astype(float)

    data.index = np.arange(data.shape[0])
    labels = data.price
    data.drop(columns=["id", "date", "zipcode", "price"], inplace=True)

    return data, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col in X.columns:
        covariance = X[col].cov(y)
        standard_deviation_x = X[col].std()
        standard_deviation_y = y.std()
        pearson_correlation = covariance / (standard_deviation_x * standard_deviation_y)
        print(f"{col}: {pearson_correlation}")

        plt.scatter(X[col], y)
        plt.title(f"{col} - Pearson Correlation: {pearson_correlation:.3f}")
        plt.xlabel(f"{col}")
        plt.ylabel("Price")
        plt.savefig(f"{output_path}/{col}.png")
        plt.close()



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, labels = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, labels, "./plots/ex2/features")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, labels, train_proportion=0.2)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    train_proportions = np.linspace(0.1, 1, 91)
    test_loss_averages = []
    test_loss_stds = []
    for train_proportion in train_proportions:
        if train_proportion * 10 == int(train_proportion * 10):
            print(f"Fitting model with {train_proportion*100}% of training data")
        proportion_losses = []
        for i in range(10):
            sampled_X = train_X.sample(frac=train_proportion)
            sampled_y = train_y.loc[sampled_X.index]
            regressor = LinearRegression(include_intercept=True)
            regressor.fit(sampled_X, sampled_y)
            proportion_losses.append(regressor.loss(test_X, test_y))
        test_loss_averages.append(np.mean(proportion_losses))
        test_loss_stds.append(np.std(proportion_losses))

    test_loss_averages = np.array(test_loss_averages)
    test_loss_stds = np.array(test_loss_stds)
    plt.plot(train_proportions * 100, test_loss_averages, label="Average loss")
    plt.fill_between(train_proportions * 100, test_loss_averages - 2 * test_loss_stds,
                     test_loss_averages + 2 * test_loss_stds, alpha=0.2)
    plt.xlabel("Training Proportion (%)")
    plt.ylabel("Average Test Loss")
    plt.title("Average Test Loss as a Function of Training Proportion")
    plt.savefig("./plots/ex2/train_proportion_vs_test_loss.png")
    plt.close()

    print("Done")
