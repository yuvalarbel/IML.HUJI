import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    ada = AdaBoost(wl=DecisionStump, iterations=n_learners)
    ada.fit(train_X, train_y)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    test_errors = q1(ada, n_learners, test_X, test_y, train_X, train_y, noise)

    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["circle", "x"])

    if noise == 0:
        # Question 2: Plotting decision surfaces
        q2(ada, test_X, test_y, lims, symbols)

        # Question 3: Decision surface of best performing ensemble
        q3(ada, test_X, test_errors, test_y, lims, symbols)

    # Question 4: Decision surface with weighted samples
    q4(ada, train_X, train_y, lims, symbols, noise)


def q1(ada, n_learners, test_X, test_y, train_X, train_y, noise):
    train_errors = []
    test_errors = []
    for i in range(n_learners):
        train_errors.append(ada.partial_loss(train_X, train_y, i + 1))
        test_errors.append(ada.partial_loss(test_X, test_y, i + 1))
    plt.plot(range(n_learners), train_errors, label='Train Error')
    plt.plot(range(n_learners), test_errors, label='Test Error')
    plt.xlabel('Number of Learners')
    plt.ylabel('Misclassification Error')
    plt.legend()
    plt.title(f'AdaBoost Training and Test Errors - Noise: {noise}')
    plt.savefig(f"./plots/ex4/adaboost_errors_noise_{str(noise).replace('.' , '_')}.png")
    plt.close()
    return test_errors


def q2(ada, test_X, test_y, lims, symbols):
    T = [5, 50, 100, 250]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Decision Boundary for T={t}" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces(get_decision_boundary_and_test_markers(lims, symbols, test_X, test_y, ada, t),
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title=f"Decision Boundaries of Ensemble up to Different Iterations (Noiseless)",
                      margin=dict(t=100), font=dict(size=16)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


def q3(ada, test_X, test_errors, test_y, lims, symbols):
    best_ensemble_size = np.argmin(test_errors) + 1
    accuracy = 1 - test_errors[best_ensemble_size - 1]
    fig = go.Figure(get_decision_boundary_and_test_markers(lims, symbols, test_X, test_y, ada, best_ensemble_size))
    fig.update_layout(title=f"Decision Boundary of Ensemble of Size {best_ensemble_size} (Noiseless) - Accuracy: {accuracy}",
                      margin=dict(t=100), font=dict(size=16)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


def q4(ada, train_X, train_y, lims, symbols, noise):
    magnitude = 20 if noise == 0 else 5
    weights = ada.D_ / np.max(ada.D_) * magnitude
    fig = go.Figure([decision_surface(ada.predict, lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                                marker=dict(size=weights, symbol=symbols[(train_y > 0).astype(int)],
                                            color=train_y, colorscale=[custom[0], custom[-1]],
                                            line=dict(color=train_y, width=1)),
                                name='Training Set')])
    fig.update_layout(title=f"Decision Boundary of Full Ensemble of Size with Weighted Samples - Noise: {noise}",
                      margin=dict(t=100), font=dict(size=16)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


def get_decision_boundary_and_test_markers(lims, symbols, test_X, test_y, ada, t):
    return [decision_surface(lambda X: ada.partial_predict(X, T=t), lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=test_y, symbol=symbols[(test_y > 0).astype(int)],
                                   colorscale=[custom[0], custom[-1]],
                                   line=dict(color="black", width=1)))]


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
    print('Done')
