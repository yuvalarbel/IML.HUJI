from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    # Draw 1000 samples from a Gaussian distribution with mean 10 and variance 1
    mu = 10
    var = 1
    full_sample_size = 1000
    X = np.random.normal(mu, var, full_sample_size)

    # Fit a Gaussian model to the data
    uvg = UnivariateGaussian(biased_var=False)
    uvg.fit(X)

    # Print the estimated expectation and variance
    print((uvg.mu_, uvg.var_))

    # Question 2 - Empirically showing sample mean is consistent
    # Get expectations over different sample sizes
    sample_sizes = np.arange(10, 1001, 10)
    expectations = np.array([UnivariateGaussian(biased_var=False).fit(X[:n]).mu_ for n in sample_sizes])
    absolute_distances = np.abs(expectations - mu)

    # Plot the absolute distance between the sample mean and the true mean
    go.Figure([go.Scatter(x=sample_sizes, y=absolute_distances, mode='markers+lines', name=r'Absolute Distance')],
              layout=go.Layout(title=r"Absolute Distance Between Estimated and True Value of Expectation",
                               xaxis_title="Sample Size",
                               yaxis_title="Absolute Distance")).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = uvg.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdfs, mode='markers', name=r'Empirical PDF')],
              layout=go.Layout(title=r"Empirical PDF of Fitted Model on Samples",
                               xaxis_title="Ordered Sample Values",
                               yaxis_title="PDFs of Samples")).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    full_sample_size = 1000

    # Draw 1000 samples from a Multivariate Gaussian distribution with mean mu and covariance cov
    X = np.random.multivariate_normal(mu, cov, full_sample_size)

    # Fit a Multivariate Gaussian model to the data
    mvg = MultivariateGaussian()
    mvg.fit(X)

    # Print the estimated expectation and covariance matrix
    print(mvg.mu_)
    print(mvg.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    f1func, f3func = np.meshgrid(f1, f3)

    # Get the log-likelihood for each point in the grid
    ll = MultivariateGaussian.log_likelihood
    get_log_likelihood = lambda x, y: ll(np.array([x, 0, y, 0], dtype=object), cov, X)
    log_likelihoods = get_log_likelihood(f1func, f3func).T

    # Plot the log-likelihoods
    go.Figure([go.Heatmap(z=log_likelihoods, x=f3, y=f1, colorscale='Viridis')],
              layout=go.Layout(title=r"Log Likelihoods of Each Value of F1 and F3",
                               xaxis_title=r"$f_3$",
                               yaxis_title=r"$f_1$")).show()

    # Question 6 - Maximum likelihood
    row, col = np.argwhere(log_likelihoods == log_likelihoods.max())[0]
    print("f1:", round(f1[row], 3))
    print("f2:", round(f3[col], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
