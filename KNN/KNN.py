import numpy as np 
from scipy import stats 
from tqdm import tqdm 

def EuclideanDistance(x1, x2) : 
    dist = np.sum((x1 - x2)**2, axis =1)
    return dist 

def ManhattanDistance(x1, x2) : 
    dist = np.sum(np.abs(x1 - x2), axis = 1)
    return dist

class KNN : 
    def __init__(self, k, distance_metric = EuclideanDistance, task_type = 'Classification') : 
        self._k = k 
        self._distance_metric = distance_metric 
        self._task_type = task_type

    def fit(self, x, y) : 
        self._x = x 
        self._y = y

    def predict(self, newExample) : 
        distance_vector = self._distance_metric(self._x, newExample)

        k_nearest_neighbours_indices = np.argpartition(distance_vector, self._k)[:self._k]

        k_nearest_neighbours = self._y[k_nearest_neighbours_indices]

        if self._task_type == 'Classification' : 
            label = stats.mode(k_nearest_neighbours)[0]

        else :
            label = k_nearest_neighbours.mean()

        return label, k_nearest_neighbours_indices

    def eval(self, x_test, y_test) : 
        if self._task_type == 'Classification' : 
            y_predicted = np.zeros(y_test.shape)
            for i in tqdm(range(y_test.shape[0])) : 
                y_predicted[i], _ = self.predict(x_test[i, :])
            error = np.mean(y_test == y_predicted, axis = 0)
        else : 
            y_predicted = np.zeros(y_test.shape)
            for i in tqdm(range(y_test.shape[0])) : 
                y_predicted[i], _ = self.predict(x_test[i, :])

            error_vector = y_predicted - y_test 
            error = np.sqrt((error_vector.T @ error_vector)/ error_vector.ravel().shape[0])

        return error 
