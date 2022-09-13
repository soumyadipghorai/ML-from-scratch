import numpy as np 
import itertools
import functools
from tqdm import tqdm 


class LeastSquareClassification(object) : 
    def __init__(self) : 
        self.t0 = 20 
        self.t1 = 1000

    def predict(self, X:np.ndarray) -> np.ndarray : 
        assert X.shape[-1] == self.w.shape[0], f"X shape {X.shape} and w shape {self.w.shape}, are not compatible"
        return np.argmax(X @ self.w, axis = 1) # class with highest lc of the features 

    def predict_internal(self, X:np.ndarray) -> np.ndarray : # for loss computation
        assert X.shape[-1] == self.w.shape[0], f"X shape {X.shape} and w shape {self.w.shape}, are not compatible"
        return X @ self.w
    
    def lossRidge(self, X: np.ndarray, y: np.ndarray, reg_rate : float) -> float : 
        e = self.predict_internal(X) -y 
        return (1/2)*(np.transpose(e) @ e) + (reg_rate/2)*(np.transpose(self.w) @ self.w)

    def rmseRidge(self, X:np.ndarray, y:np.ndarray, reg_rate:float) -> float : 
        return np.sqrt((2/X.shape[0]) * self.lossRidge(X, y, reg_rate))

    def fitMulti(self, X: np.ndarray, y:np.ndarray, reg_rate:float) -> np.ndarray : 
        self.w = np.zeros((X.shape[1], y.shape[1]))
        eye = np.eye(np.size(X, 1))
        self.w = np.linalg.solve(
            reg_rate * eye + X.T @ X, 
            X.T @ y,
        )
        return self.w

    def calculate_ridge_gradient(self, X: np.ndarray, y: np.ndarray, reg_rate:float) : 
        return (np.transpose(X) @ (self.predict_internal(X) - y)) + (reg_rate * self.w)

    def update_weight(self, grad, lr:float) : 
        return (self.w - lr*grad)

    def learning_schedule(self, t) : 
        return self.t0 / (t + self.t1)

    def gradient_descent(self, X: np.ndarray, y : np.ndarray, num_epochs : int, lr:float, reg_rate:float) -> np.ndarray: 
        
        self.w = np.zeros((X.shape[1], y.shape[1]))
        self.w_all = []
        self.err_all = []
        
        for i in tqdm(np.arange(0, num_epochs)) : 
            dJdW = self.calculate_ridge_gradient(X, y, reg_rate)
            self.w_all.append(self.w)
            self.err_all.append(self.lossRidge(X, y, reg_rate))
            self.w = self.update_weight(dJdW, lr)

        return self.w 

    def sgd(self, X:np.ndarray, y:np.ndarray, num_epochs : int, reg_rate:float) -> np.ndarray: 
        
        self.w = np.zeros((X.shape[1], y.shape[1]))
        self.w_all = []
        self.err_all = []
        
        for epoch in tqdm(range(num_epochs)) : 
            for i in range(X.shape[0]) : 
                random_index = np.random.randint(X.shape[0])
                xi = X[random_index : random_index+1]
                yi = y[random_index: random_index+1]

                self.w_all.append(self.w)
                self.err_all.append(self.lossRidge(xi, yi, reg_rate))

                gradient = 2 * self.calculate_ridge_gradient(xi, yi, reg_rate)
                lr = self.learning_schedule(epoch * X.shape[0] + i)
                self.w = self.update_weight(gradient, lr)

        return self.w