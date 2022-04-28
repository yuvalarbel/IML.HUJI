from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.metrics import accuracy
from typing import Tuple
from utils import *
from math import atan2, pi
import os
import matplotlib.pyplot as plt


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "../datasets/linearly_separable.npy"),
                 ("Linearly Inseparable", "../datasets/linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron_callback = lambda fit, x_i, y_i: losses.append(fit.loss(X, y))
        perceptron = Perceptron(callback=perceptron_callback)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        plt.plot(losses, label=n)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(f"Perceptron Loss as a Function of Iterations - {n} Dataset")
        plt.savefig(f"./plots/ex3/perceptron_loss_{'_'.join(n.lower().split())}.png")
        plt.close()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return mu[0] + xs, mu[1] + ys


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["../datasets/gaussian1.npy", "../datasets/gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        gnb_pred = gnb.predict(X)

        lda = LDA()
        lda.fit(X, y)
        lda_pred = lda.predict(X)

        # Add traces for data-points setting symbols and colors
        dataset = os.path.splitext(os.path.basename(f))[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        markers = "oxv"
        colors = "rgb"

        for i, true_class in enumerate(np.unique(y)):
            shape = markers[i]
            for j, pred_class in enumerate(np.unique(y)):
                color = colors[j]

                gnb_inds = np.logical_and(y == true_class, gnb_pred == pred_class)
                lda_inds = np.logical_and(y == true_class, lda_pred == pred_class)

                ax1.scatter(X[gnb_inds, 0], X[gnb_inds, 1],
                            marker=shape, c=color, label=f"{true_class}-{pred_class}-GNB")
                ax2.scatter(X[lda_inds, 0], X[lda_inds, 1],
                            marker=shape, c=color, label=f"{true_class}-{pred_class}-LDA")

        # Add `X` dots specifying fitted Gaussians' means
        ax1.scatter(gnb.mu_[:, 0],
                    gnb.mu_[:, 1],
                    marker="X", c="black", s=100)
        ax2.scatter(lda.mu_[:, 0], lda.mu_[:, 1],
                    marker="X", c="black", s=100)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i, true_class in enumerate(np.unique(y)):
            ax1.plot(*get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])),
                     color="black")
            ax2.plot(*get_ellipse(lda.mu_[i], lda.cov_), color="black")

        ax1.set_title(f"Gaussian Naive Bayes - Accuracy: {round(accuracy(y, gnb_pred), 2)}")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax2.set_title(f"LDA - Accuracy: {round(accuracy(y, lda_pred), 3)}")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")

        fig.suptitle(f"Gaussian Naive Bayes vs LDA - {dataset} Dataset")
        fig.savefig(f"./plots/ex3/gaussian_naive_bayes_vs_lda_{dataset}.png")
        plt.close(fig)

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
