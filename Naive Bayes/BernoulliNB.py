import numpy as np 
np.random.seed(1234)
import itertools
import functools
from tqdm import tqdm 

class BernoulliNB(object) : 
    def __init__(self, alpha = 1.0) : 
        self.alpha = alpha 

    def fit(self, X, y) : 
        n_samples, n_features = X.shape 
        class_count = np.unique(y)
        n_classes = len(class_count)

        self.w = np.zeros((n_classes, n_features), dtype = np.float64)
        self.w_priors = np.zeros(n_classes, dtype = np.float64)

        for c in tqdm(range(n_classes), colour = 'CYAN') : 
            X_c = X[y == c]

            self.w[c, :] = (np.sum(X_c, axis = 0) +  + self.alpha)/(X_c.shape[0] + 2 * self.alpha)

            self.w_priors[c] = (X_c.shape[0] + self.alpha)/(float(n_samples) + n_classes * self.alpha)

        print('class conditional density: ',self. w)
        print('prior: ', self.w_priors)

    def log_likelihood_prior_prob(self, X) : 
        return X @ (np.log(self.w).T) + (1 - X)@ np.log((1-self.w).T) + np.log(self.w_priors)

    def predict_probs(self, X) : 
        q = self.log_likelihood_prior_prob(X)
        return np.exp(q) / np.expand_dims(np.sum(np.exp(q), axis = 1), axis = 1)

    def predict(self, X) : 
        return np.argmax(self.log_likelihood_prior_prob(X), axis = 1)