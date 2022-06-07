from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import matplotlib.pyplot as plt


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    print(f"\nRunning polynomial degree selection for {n_samples} samples with noise {noise}")

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    epsilon = np.random.normal(0, noise, n_samples)
    response = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.linspace(-1.2, 2, n_samples)
    y_true = response(X)
    y_noise = response(X) + epsilon

    X_df = pd.DataFrame(X, columns=['x'])
    y_series = pd.Series(y_noise, name='y')

    X_train, y_train, X_test, y_test = split_train_test(X_df, y_series, train_proportion=2/3)
    X_train, X_test = X_train['x'].to_numpy(), X_test['x'].to_numpy()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    plt.plot(X, y_true, 'k-', label='True Function')
    plt.scatter(X_train, y_train, c='b', label='Training Data')
    plt.scatter(X_test, y_test, c='r', label='Test Data')
    plt.xlabel('$x$')
    plt.ylabel(r'$f\left(x\right)$')
    plt.title(fr'Polynomial Regression for Data with Noise: $\varepsilon\sim\mathcal{{N}}\left(0,\sigma^{{2}}={noise}\right)$')
    plt.legend()
    plt.savefig(f'plots/ex5/polynomial_regression_data_samples_{n_samples}_noise_{noise}.png')
    plt.close()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    errors = []
    for degree in range(11):
        model = PolynomialFitting(k=degree)
        errors.append(cross_validate(model, X_train, y_train,
                                     scoring=mean_square_error, cv=5))

    train_errors, validation_errors = zip(*errors)

    # Plot for each value of k the average training- and validation errors
    plt.plot(range(11), train_errors, label='Training Error', marker='o', c='b')
    plt.plot(range(11), validation_errors, label='Validation Error', marker='o', c='r')
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Mean Square Error')
    plt.title(fr'Polynomial Regression Error for Degrees 0,1,...,10 with Noise: $\sigma^{{2}}={noise}$')
    plt.legend()
    plt.savefig(f'plots/ex5/polynomial_regression_errors_samples_{n_samples}_noise_{noise}.png')
    plt.close()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_degree = int(np.argmin(validation_errors))
    print(f'Best degree: {best_degree}')
    print(f'Best validation error: {validation_errors[best_degree]} ({round(validation_errors[best_degree], 2)})')
    model = PolynomialFitting(k=best_degree)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_error = mean_square_error(y_test, y_pred)
    print(f'Test error: {test_error} ({round(test_error, 2)})')

    if n_samples == 1500:
        model = PolynomialFitting(k=4)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_error = mean_square_error(y_test, y_pred)
        print(f'Test error for polynomial degree 4: {test_error} ({round(test_error, 2)})')


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    X_train, y_train = X.iloc[:n_samples, :].to_numpy(), y.iloc[:n_samples].to_numpy()
    X_test, y_test = X.iloc[n_samples:, :].to_numpy(), y.iloc[n_samples:].to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lamda_values = np.linspace(0.0001, 1, n_evaluations)
    ridge_errors = []
    lasso_errors = []

    for lam in lamda_values:
        model = RidgeRegression(lam=lam)
        ridge_errors.append(cross_validate(model, X_train, y_train,
                                           scoring=mean_square_error, cv=5))
        model = Lasso(alpha=lam)
        lasso_errors.append(cross_validate(model, X_train, y_train,
                                           scoring=mean_square_error, cv=5))

    ridge_train_errors, ridge_validation_errors = zip(*ridge_errors)
    lasso_train_errors, lasso_validation_errors = zip(*lasso_errors)

    plt.plot(lamda_values, ridge_train_errors, label='Ridge Training Error',
             marker='o', c='b', markersize=3, linewidth=0.5)
    plt.plot(lamda_values, ridge_validation_errors, label='Ridge Validation Error',
             marker='o', c='r', markersize=3, linewidth=0.5)
    plt.xlabel('Value of Regularization Parameter')
    plt.ylabel('Mean Square Error')
    plt.title(fr'Ridge Regression Error for Regularization Parameter Values')
    plt.legend()
    plt.savefig(f'plots/ex5/ridge_regression_errors.png')
    plt.close()

    plt.plot(lamda_values, lasso_train_errors, label='Lasso Training Error',
             marker='o', c='b', markersize=3, linewidth=0.5)
    plt.plot(lamda_values, lasso_validation_errors, label='Lasso Validation Error',
             marker='o', c='r', markersize=3, linewidth=0.5)
    plt.xlabel('Value of Regularization Parameter')
    plt.ylabel('Mean Square Error')
    plt.title(fr'Lasso Regression Error for Regularization Parameter Values')
    plt.legend()
    plt.savefig(f'plots/ex5/lasso_regression_errors.png')
    plt.close()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    reg_value_ridge = lamda_values[np.argmin(ridge_validation_errors)]
    reg_value_lasso = lamda_values[np.argmin(lasso_validation_errors)]

    print(f'\n\nBest Ridge Regularization Parameter: {reg_value_ridge}')
    print(f'Best Lasso Regularization Parameter: {reg_value_lasso}')

    ridge_model = RidgeRegression(lam=reg_value_ridge)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    ridge_test_error = mean_square_error(y_test, y_pred)
    print(f'Ridge Test Error: {ridge_test_error} ({round(ridge_test_error, 2)})')

    lasso_model = Lasso(alpha=reg_value_lasso)
    lasso_model.fit(X_train, y_train)
    y_pred = lasso_model.predict(X_test)
    lasso_test_error = mean_square_error(y_test, y_pred)
    print(f'Lasso Test Error: {lasso_test_error} ({round(lasso_test_error, 2)})')

    least_squares_model = LinearRegression()
    least_squares_model.fit(X_train, y_train)
    y_pred = least_squares_model.predict(X_test)
    least_squares_test_error = mean_square_error(y_test, y_pred)
    print(f'Least Squares Test Error: {least_squares_test_error} ({round(least_squares_test_error, 2)})')


if __name__ == '__main__':
    np.random.seed(0)

    for n_samples, noise in ((100, 5), (100, 0), (1500, 10)):
        select_polynomial_degree(n_samples, noise)

    select_regularization_parameter()
    print("\nDone")
