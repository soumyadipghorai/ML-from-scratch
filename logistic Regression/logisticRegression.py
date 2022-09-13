import numpy as np 
np.random.seed(1234)
import itertools
import functools
from tqdm import tqdm 

class LogisticRegression(object):

    '''Logistic Regression model

        y = sigmoid(X @ w)
    '''
    
    def set_weight_vector(self, w):
        self.w = w
    
    def linear_combination(self, X:np.ndarray) -> np.ndarray:
        '''Calculates linear combination of features.
    
        The linear combination is calculated witht he following vectorised form

        z = Xw
        Args:
            X: feature matrix with shape(n, m)
            w: weight vector with shape(m,)

        Returns:
            Linear combination of features with shape (n,)
        
        '''
        return X @ self.w

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        '''Calculates sigmoid of linear combination of features z
    
         Args:
            z: list of floats

            Returns:
                 Sigmoid function of linear combination of features as an array
        '''

        return 1/(1 + np.exp(-z))
    
    def activation(self, X:np.ndarray) -> np.ndarray:
        '''Calculates sigmoid activation for logistic regression.
        
        Args:
            X: Feature matrix with shape (n, m)

        Returns:
            activation vector with shape (n,)
        '''
        
        return self.sigmoid(self.linear_combination(X))

    def predict(self, X: np.ndarray, threshold=0.5) -> np.ndarray:
        '''Predicts class label for samples

        Args:
            X: feature matrix with shape(n, m)
            w: weight vector with shape(m,)
            threshold: Probability the=reshold for prediction
    
        Returns:
            Predicted class labels
        '''
        return np.where(self.activation(X) > threshold, 1, 0).astype(int)

    def loss(self, X:np.ndarray, y:np.ndarray, reg_rate:float) -> float:
        '''Calculate loss function for a given weight vactor

        Args:
            X: feature matrix with shape(n, m)
            y: label vector with shape(n,)
            w: weight vector with shape(m,)
            reg_rate: L2 regularisation rate

        Returns:
            Loss function
        '''
        predicted_prob = self.activation(X)
        return (-1 * (np.sum(y @ np.log(predicted_prob)) + (1 - y) @ np.log(1 - predicted_prob)))  + reg_rate * np.dot(self.w.T, self.w)

    def calculate_gradient(self, X:np.ndarray, y:np.ndarray, reg_rate: float) -> np.ndarray:
        '''Calculates gradients of loss function wrt weight vector on training set

        Args: 
            X: Feature matrix for training data.
            y:Label vector for training data.
            reg_rate: regularisation rate

        Returns:
            A vector of gradients
        '''

        return X.T @ (self.activation(X) - y) + reg_rate * self.w

    def update_weights(self, grad:np.ndarray, lr:float) -> np.ndarray:
        '''Updates the weights based on the gradient of loss function
        Args:
            1. w: Weight vector
            2. grad: gradient of loss w.r.t w
            3.  lr: learning rate
        Returns:
            Updated weights
        '''
        return (self.w - lr*grad)

    def gd(self, X:np.ndarray, y:np.ndarray, num_epochs:int, lr:float, reg_rate:float) -> np.ndarray:
        '''Estimates the parameters of logistic regression model with gradient descent'
    
        Args:
            X: Feaature matrix for training data.
            y: Label vector for traaining data.
            num_epochs: NUmber of training steps
            lr: Learning rate
            reg_rate: Regularisation rate

        Returns:
            Weight vector: Final weight vector
        '''

        self.w = np.zeros(X.shape[1])
        self.w_all = []
        self.err_all = []

        for i in tqdm(np.arange(0, num_epochs)):
            
            dJdW = self.calculate_gradient(X, y, reg_rate)
            self.w_all.append(self.w)
            self.err_all.append(self.loss(X, y, reg_rate))
            self.w = self.update_weights(dJdW, lr)

        return self.w 
