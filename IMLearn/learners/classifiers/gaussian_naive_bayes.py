from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # get problem dimensions
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
        self.classes_ = np.unique(y)
        num_classes = self.classes_.shape[0]
        num_features = X.shape[1]
        num_samples = X.shape[0]

        # aggregate data
        agg = np.zeros(num_classes)
        mus = np.zeros((num_classes, num_features))
        vars = np.zeros((num_classes, num_features))
        for i, c in enumerate(self.classes_):
            cur_ind = y == c
            agg[i] = len(y[cur_ind])
            mus[i] = np.sum(X[cur_ind], axis=0) / agg[i]
            vars[i] = np.sum((X[cur_ind] - mus[i]) ** 2, axis=0) / (agg[i] - 1)

        self.pi_ = agg / num_samples
        self.mu_ = mus
        self.vars_ = vars

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
        likelihood = self.likelihood(X)
        return self.classes_[np.argmax(likelihood * self.pi_, axis=1)]

    def create_likelihood_func(self, X):
        """
        Creates a likelihood calculation function given the test data.
        """
        num_features = X.shape[1]

        def func(i, k):
            centered_X = X[i] - self.mu_[k]
            mahalanobis = np.einsum('nbi,bi,nbi->nb', centered_X, self.vars_[k][0] ** (-1), centered_X)
            return np.exp(-.5 * mahalanobis) / np.sqrt((2 * np.pi ** num_features) * np.prod(self.vars_[k]))

        return func

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
        num_classes = self.classes_.shape[0]
        num_samples = X.shape[0]
        u, v = np.meshgrid(np.arange(num_samples), np.arange(num_classes), indexing='ij', sparse=True)
        func = self.create_likelihood_func(X)
        return func(u, v)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
