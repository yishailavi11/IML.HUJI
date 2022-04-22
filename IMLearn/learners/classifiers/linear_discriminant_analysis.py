from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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
        cov = np.zeros((num_features, num_features))
        for i, c in enumerate(self.classes_):
            cur_ind = y == c
            agg[i] = len(y[cur_ind])
            mus[i] = np.sum(X[cur_ind], axis=0) / agg[i]
        for i in range(num_samples):
            centered_xi = X[i] - mus[y[i]]
            cov = cov + (centered_xi.reshape(-1, 1) @ centered_xi.reshape(1, -1))

        # assign data to fields
        self.pi_ = agg / num_samples
        self.mu_ = mus
        self.cov_ = cov / (num_samples - num_classes)
        self._cov_inv = inv(self.cov_)

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
        return self.classes_[np.argmax(likelihood*self.pi_, axis=1)]

    def create_likelihood_func(self, X):
        """
        Creates a likelihood calculation function given the test data.
        """
        num_features = X.shape[1]

        def func(i, k):
            centered_X = X[i] - self.mu_[k]
            mahalanobis = np.einsum('nbi,ij,nbj->nb', centered_X, self._cov_inv, centered_X)
            return np.exp(-.5 * mahalanobis) / np.sqrt(((2 * np.pi) ** num_features) * det(self.cov_))

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
        u, v = np.meshgrid(np.arange(num_samples), np.arange(num_classes), sparse=True, indexing='ij')
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
