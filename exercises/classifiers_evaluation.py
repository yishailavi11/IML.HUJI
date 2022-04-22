from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


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
    for n, f in [("Linearly Separable", "..\\datasets\\linearly_separable.npy"),
                 ("Linearly Inseparable", "..\\datasets\\linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def calc_loss_func(fit: Perceptron, cur_x: np.ndarray, cur_y: int):
            losses.append(fit.loss(X, y))

        perceptron = Perceptron(callback=calc_loss_func)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(losses), dtype=int), losses)
        plt.xlabel('Number of iterations')
        plt.ylabel('Misclassification error')
        plt.title(f'Perceptron loss as a function of the number of iterations- {n}')
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray, ax):
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

    return ax.scatter(mu[0] + xs, mu[1] + ys, c='black', s=2)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["..\\datasets\\gaussian1.npy", "..\\datasets\\gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda_classifier = LDA()
        ng_classifier = GaussianNaiveBayes()
        lda_classifier.fit(X, y)
        ng_classifier.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        # create data for plotting
        y_pred_ng = ng_classifier.predict(X)
        y_pred_lda = lda_classifier.predict(X)
        ng_acc = accuracy(y, y_pred_ng)
        lda_acc = accuracy(y, y_pred_lda)

        # scatter data
        plt.set_cmap('gist_rainbow')
        cmap = plt.get_cmap()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        markers = ['o', '^', 's']
        for i in range(3):
            ax1.scatter(X[y == i, 0], X[y == i, 1], c=cmap(y_pred_ng[y == i]*0.5), marker=markers[i])
            ax2.scatter(X[y == i, 0], X[y == i, 1], c=cmap(y_pred_lda[y == i]*0.5), marker=markers[i])
        ax1.set_title(f'Gaussian Naive Bayes Classifier\nAccuracy: {round(ng_acc, 5)}')
        ax2.set_title(f'LDA Classifier\nAccuracy: {round(lda_acc, 5)}')

        # add legend
        legend_elements = []
        legend_titles = []
        for i in range(3):
            legend_elements += [Line2D([0], [0], color=cmap(i * 0.5), lw=4),
                                Line2D([0], [0], color='black', marker=markers[i])]
            legend_titles += [f'Predicted class {i}', f'True class {i}']
        fig.legend(legend_elements, legend_titles, loc='lower right')

        # Add `X` dots specifying fitted Gaussians' means
        ng_means = ng_classifier.mu_
        lda_means = lda_classifier.mu_
        ax1.scatter(ng_means[:, 0], ng_means[:, 1], c='black', marker='x', s=70)
        ax2.scatter(lda_means[:, 0], lda_means[:, 1], c='black', marker='x', s=70)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(3):
            get_ellipse(ng_means[i], np.diag(ng_classifier.vars_[i]), ax1)
            get_ellipse(lda_means[i], lda_classifier.cov_, ax2)

        plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
