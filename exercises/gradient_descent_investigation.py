import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.metrics.loss_functions import misclassification_error
from IMLearn.model_selection.cross_validate import cross_validate
from utils import *

import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"
c = [custom[0], custom[-1]]


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_ = []
    weights_ = []

    def callback(solver, weights, val, grad, t, eta, delta):
        values_.append(val.copy())
        weights_.append(weights.copy())

    return callback, values_, weights_


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    convergences = {}
    for module in (L1, L2):
        convergences[module] = []
        for eta in etas:
            label = f"{module.__name__} Model, with eta={eta}"
            callback, values, weights = get_gd_state_recorder_callback()
            solver = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            solution = solver.fit(module(weights=init.copy()), X=np.array([]), y=np.array([]))
            convergences[module].append((np.array(values), eta))
            plot_descent_path(module, np.array(weights), title=label).show()

    for module in (L1, L2):
        go.Figure([go.Scatter(x=np.arange(vals.size) + 1, y=vals, mode="markers+lines", name=f"Eta={eta}")
                   for vals, eta in convergences[module]],
                  layout=go.Layout(title=f"Convergence Rates of {module.__name__} Module",
                                   xaxis_title=f"Iteration #",
                                   yaxis_title=f"{module.__name__} Norm Value")).show()

    for module in (L1, L2):
        for vals, eta in convergences[module]:
            print(f"Lowest {module.__name__} Norm Value with Eta={eta}: {vals.min()}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    convergences = []
    for gamma in gammas:
        label = f"L1 Model, with eta={eta} and gamma={gamma}"
        callback, values, weights = get_gd_state_recorder_callback()
        solver = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback)
        solution = solver.fit(L1(weights=init.copy()), X=np.array([]), y=np.array([]))
        values, weights = np.array(values), np.array(weights)
        convergences.append((values, gamma))
        print(f"Lowest L1 Norm Value with Eta={eta} and Gamma={gamma}: {values.min()}")

    # Plot algorithm's convergence for the different values of gamma
    go.Figure([go.Scatter(x=np.arange(vals.size) + 1, y=vals, mode="markers+lines", name=f"Gamma={gamma}")
               for vals, gamma in convergences],
              layout=go.Layout(title=f"Convergence Rates of L1 Module with Eta={eta}",
                               xaxis_title=f"Iteration #",
                               yaxis_title=f"L1 Norm Value")).show()

    # Plot descent path for gamma=0.95
    gamma = 0.95
    for module in (L1, L2):
        label = f"{module.__name__} Model, with eta={eta} and gamma={gamma}"
        callback, values, weights = get_gd_state_recorder_callback()
        solver = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=callback)
        solution = solver.fit(module(weights=init.copy()), X=np.array([]), y=np.array([]))
        values, weights = np.array(values), np.array(weights)
        plot_descent_path(module, weights, title=label).show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    lr = 1e-4
    max_iter = 20000
    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    train_predictions = lg.predict_proba(X_train)

    fpr, tpr, thresholds = roc_curve(y_train, train_predictions)
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    opt_threshold = thresholds[np.argmax(tpr - fpr)]
    print(f"Max Threshold: {opt_threshold}")

    pred_test = (lg.predict_proba(X_test) >= opt_threshold).astype(int)
    print(f"Test Error with Pre-Trained Model: {misclassification_error(y_test, pred_test)}")

    lg_star = LogisticRegression(alpha=opt_threshold)
    lg_star.fit(X_train, y_train)
    test_error = lg_star.loss(X_test, y_test)
    print(f"Test Error with Retrained Model: {test_error}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    optimize_regularized_logistic_regression(X_test, X_train, y_test, y_train, "l1")
    optimize_regularized_logistic_regression(X_test, X_train, y_test, y_train, "l2")


def optimize_regularized_logistic_regression(X_test, X_train, y_test, y_train, penalty,
                                             lamdas=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]):
    scores = []
    for lam in lamdas:
        scores.append(cross_validate(LogisticRegression(penalty=penalty, lam=lam),
                                     X_train.to_numpy(), y_train.to_numpy(), misclassification_error))
    train_errors, validation_errors = zip(*scores)
    best_lamda = lamdas[np.argmin(validation_errors)]
    print(f"{penalty.capitalize()} Regularization Best Lambda: {best_lamda}")
    reg_lg = LogisticRegression(penalty=penalty, lam=best_lamda)
    reg_lg.fit(X_train, y_train)
    print(f"Test Error with {penalty.capitalize()} Regularization: {reg_lg.loss(X_test, y_test)}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
    print("Done")
