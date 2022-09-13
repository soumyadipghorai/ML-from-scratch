import numpy as np 
np.random.seed(1234)
import itertools
import functools
from tqdm import tqdm 

class GaussianNB(object):
    def fit(self, X, y):
        '''Parameter estimation for Gaussian NB'''

        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialise mean, var, and prior for each class.
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in tqdm(enumerate(self._classes)):

            #Get examples with label y=c
            X_c = X[y == c]

            #Estimate mean from the training examples of class c.
            self._mean[idx, :] = X_c.mean(axis=0)

            #Estimate variance from the training examples of class c.
            self._var[idx, :] = X_c.var(axis=0)

            #Estimate priors.
            self._priors[idx] = X_c.shape[0] / float(n_samples)

        print("Mean: ", self._mean)
        print("Variance: ", self._var)
        print("Priors: ", self._priors)

    def _calc_pdf(self, class_idx, X):
        '''Calculates probability density for samples for class label class_idx'''

        mean = self._mean[class_idx]
        var = np.diag(self._var[class_idx])
        z = np.power(2 * np.pi, X.shape[0]/2) * np.power(np.linalg.det(var), 1/2)
        return (1/z) * np.exp(-(1/2)*(X - mean).T @ (np.linalg.inv(var)) @ (X - mean))

    def _calc_prod_likelihood_prior(self, X):
        '''Calculates product of likelihood and priors.'''

        self._prod_likelihood_prior = np.zeros((X.shape[0], len(self._classes)), dtype=np.float64)

        for x_idx, x in tqdm(enumerate(X)):
            for idx, c in enumerate(self._classes):
                self._prod_likelihood_prior[x_idx, c] = (np.log(self._calc_pdf(idx, x)) + np.log(self._priors[idx]))

    def predict(self, X):
        '''Predicts class labels for each example'''
        
        self._calc_prod_likelihood_prior(X)

        return np.argmax(self._prod_likelihood_prior, axis=1)

    def predict_proba(self, X):
        '''Calculates probability of each example belonging to different classes.'''

        self._calc_prod_likelihood_prior(X)

        return np.exp(self._prod_likelihood_prior) / np.expand_dims(np.sum(np.exp(self._prod_likelihood_prior), axis = 1), axis = 1)
