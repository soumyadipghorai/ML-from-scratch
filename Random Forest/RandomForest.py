import numpy as np
import pandas as pd
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm 

def bag(X, y):
    n_samples = X.shape[0]
    #Generate a random indices for a sample from the input
    indices = np.random.choice(n_samples, size = n_samples, replace=True)
    return X[indices], y[indices]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, max_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        
        for _ in tqdm(range(self.n_trees)):
            tree = DecisionTreeClassifier(
                    min_samples_split = self.min_samples_split,
                    max_depth = self.max_depth,
                    max_features= self.max_features)
            
            X_sample, y_sample = bag(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predict = np.array([tree.predict(X) for tree in self.trees])
        tree_predict = np.swapaxes(tree_predict, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_predict]
        return np.array(y_pred)