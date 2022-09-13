import numpy as np 
np.random.seed(1234)
import itertools
import functools
from tqdm import tqdm 


class MultinomialNB(object):
    def fit(self, X, y, alpha = 1):
        '''Parameter estimation for Gaussian NB'''

        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self.w = np.zeros((n_classes, n_features), dtype=np.float64)
        self.w_priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in tqdm(enumerate(self._classes)):

            X_c = X[y == c]

            total_count = np.sum(np.sum(X_c, axis = 1))

            self.w[idx, :] = (np.sum(X_c, axis = 0) + alpha)/(
                total_count + alpha * n_features
            )

            self.w_priors[idx] = (X_c.shape[0] + alpha) / float(
                n_samples + alpha * n_classes
            )

    def log_likelihood_prior_prod(self, X):
        return X @ (np.log(self.w).T) + np.log(self.w_priors)

    def predict(self, X):
        return np.argmax(self.log_likelihood_prior_prod(X), axis=1)

    def predict_proba(self, X):
        q = self.log_likelihood_prior_prod(X)
        return np.exp(q)/np.expand_dims(np.sum(np.exp(q), axis = 1), axis = 1)