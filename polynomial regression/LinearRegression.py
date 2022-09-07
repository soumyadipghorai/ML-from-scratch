import numpy as np 


class LinearRegression : 
    def __init__(self) : 
        self.t0 = 200
        self.t1 = 100000

    # def add_dummy_feature(self, X: np.ndarray) -> np.ndarray: 
    #     return np.column_stack((np.ones(X.shape[0]), X))

    def predict(self, X: np.ndarray)-> np.ndarray : 
        assert X.shape[-1] == self.w.shape[0]
        return X@self.w

    def loss(self, X: np.ndarray, y: np.ndarray) -> float : 
        e = self.predict(X) -y 
        return (1/2)*(np.transpose(e) @ e)

    def lossRidge(self, X: np.ndarray, y: np.ndarray, reg_rate : float) -> float : 
        e = self.predict(X) -y 
        return (1/2)*(np.transpose(e) @ e) + (reg_rate/2)*(np.transpose(self.w) @ self.w)

    def rmse(self, X:np.ndarray, y:np.ndarray) -> float : 
        return np.sqrt((2/X.shape[0]) * self.loss(X, y))

    def rmseRidge(self, X:np.ndarray, y:np.ndarray, reg_rate:float) -> float : 
        return np.sqrt((2/X.shape[0]) * self.lossRidge(X, y, reg_rate))

    def fit(self, X: np.ndarray, y:np.ndarray) -> np.ndarray : 
        self.w = np.linalg.pinv(X) @ y 
        return self.w 

    def fitMulti(self, X: np.ndarray, y:np.ndarray, reg_rate:float) -> np.ndarray : 
        self.w = np.zeros((X.shape[1], y.shape[1]))
        eye = np.eye(np.size(X, 1))
        self.w = np.linalg.solve(
            reg_rate * eye + X.T @ X, 
            X.T @ y,
        )
        return self.w


    def fitRidge(self, X: np.ndarray, y:np.ndarray, reg_rate:float) -> np.ndarray : 
        self.w = np.zeros((X.shape[1]))
        eye = np.eye(np.size(X, 1))
        self.w = np.linalg.solve(
            reg_rate * eye + X.T @ X, 
            X.T @ y,
        )
        return self.w

    def calculate_gradient(self, X: np.ndarray, y: np.ndarray) : 
        return np.transpose(X) @ (self.predict(X, self.w) - y)

    def calculate_ridge_gradient(self, X: np.ndarray, y: np.ndarray, reg_rate:float) : 
        return np.transpose(X) @ (self.predict(X) - y) + reg_rate * self.w

    def update_weight(self, grad, lr:float) : 
        return (self.w - lr*grad)

    # both sgd & mbgd have dynamic learning rate which is defined by these 
    def learning_schedule(self, t) : 
        return self.t0 / (t + self.t1)

    def gradient_descent(self, X: np.ndarray, y : np.ndarray, num_epochs : int, lr:float) -> np.ndarray: 
        
        self.w = np.zeros((X.shape[1]))
        self.w_all = []
        self.err_all = []
        
        for i in np.arange(0, num_epochs) : 
            dJdW = self.calculate_gradient(X, y)
            self.w_all.append(self.w)
            self.err_all.append(self.loss(X, y))
            self.w = self.update_weight(dJdW, lr)

        return self.w 

    def gradient_descent_multi(self, X: np.ndarray, y : np.ndarray, num_epochs : int, lr:float) -> np.ndarray: 
        
        self.w = np.zeros((X.shape[1], y.shape[1]))
        self.w_all = []
        self.err_all = []
        
        for i in np.arange(0, num_epochs) : 
            dJdW = self.calculate_gradient(X, y)
            self.w_all.append(self.w)
            self.err_all.append(self.loss(X, y))
            self.w = self.update_weight(dJdW, lr)

        return self.w 

    def mbgd(self, X:np.ndarray, y:np.ndarray, num_epochs : int, batch_size : int) -> np.ndarray: 
        self.w = np.zeros((X.shape[1]))
        self.w_all = []
        self.err_all = []
        mini_batch_id = 0 

        for epoch in range(num_epochs) : 
            shuffled_indices = np.random.permutation(X.shape[0])    
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            for i in range(0, X.shape[0], batch_size) : 
                mini_batch_id += 1 
                xi = X_shuffled[i:i+batch_size]
                yi = y_shuffled[i:i+batch_size]

                self.w_all.append(self.w)
                self.err_all.append(self.loss(xi, yi))

                dJdW = 2/batch_size * self.calculate_gradient(xi, yi)
                self.w = self.update_weight(dJdW, self.learning_schedule(mini_batch_id))

            return self.w

    def sgd(self, X:np.ndarray, y:np.ndarray, num_epochs : int) -> np.ndarray: 
        self.w = np.zeros((X.shape[1]))
        self.w_all = []
        self.err_all = []
        
        for epoch in range(num_epochs) : 
            for i in range(X.shape[0]) : 
                random_index = np.random.randint(X.shape[0])
                xi = X[random_index : random_index+1]
                yi = y[random_index: random_index+1]

                self.w_all.append(self.w)
                self.err_all.append(self.loss(xi, yi))

                gradient = 2 * self.calculate_gradient(xi, yi)
                lr = self.learning_schedule(epoch * X.shape[0] + i)
                self.w = self.update_weight(gradient, lr)

        return self.w

