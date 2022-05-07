import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from IMLearn.metrics.loss_functions import accuracy


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


def decision_surface(predict, t, xrange, yrange, X, y, density=120, figsize=(10, 8), title='', size=None):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel('Second feature')
    plt.ylabel('First feature')
    plt.contourf(xx, yy, pred, cmap='cool')
    cmap = 'gray'
    if size is not None:
        plt.scatter(X[:, 0], X[:, 1], cmap=cmap, c=y, s=size)
    else:
        plt.scatter(X[:, 0], X[:, 1], cmap=cmap, c=y, s=10)
    plt.show()


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adb = AdaBoost(lambda: DecisionStump(), n_learners).fit(train_X, train_y)
    learners_range_gen = range(1, n_learners)
    loss_train = [adb.partial_loss(train_X, train_y, i) for i in learners_range_gen]
    loss_test = [adb.partial_loss(test_X, test_y, i) for i in learners_range_gen]
    axis = list(learners_range_gen)
    plt.figure(figsize=(20, 6))
    plt.plot(axis, loss_train, label='Train error')
    plt.plot(axis, loss_test, label='Test error')
    plt.xlabel('Number of learners')
    plt.ylabel('Loss')
    plt.title('Adaboost loss as a function of the number of learners')
    plt.legend()
    plt.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T
    for t in T:
        decision_surface(lambda x: adb.partial_predict(x, t), t, lims[0], lims[1], test_X, test_y,
                         title=f'Adaboost decision boundaries with scattered test data after {t} iterations.')

    # Question 3: Decision surface of best performing ensemble
    best_ens = int(np.argmin(loss_test))
    print(best_ens)
    decision_surface(lambda x: adb.partial_predict(x, best_ens), best_ens, lims[0], lims[1], test_X, test_y,
                     title=f'Adaboost decision boundaries with scattered test data after {best_ens} iterations.\n'
                           f'Accuracy: {accuracy(test_y, adb.partial_predict(test_X, best_ens))}')

    # Question 4: Decision surface with weighted samples
    point_size = (adb.D_/np.max(adb.D_)*20)
    decision_surface(lambda x: adb.partial_predict(x, n_learners), n_learners, lims[0], lims[1], train_X, train_y,
                     title=f'Adaboost decision boundaries with scattered test data after {n_learners} iterations.\n'
                           f'Accuracy: {accuracy(train_y, adb.partial_predict(train_X, n_learners))}', size=point_size)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, 250)
    fit_and_evaluate_adaboost(0.4, 250)
