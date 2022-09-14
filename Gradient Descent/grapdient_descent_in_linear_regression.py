import numpy as np 
import matplotlib.pyplot as plt 

class LinearRegression : 
    
    def __init__(self, X, Y) : 
        self.X = X
        self.Y = Y 
        self.b = [0, 0]


    def update_coefficients(self, learning_rate) : 
        Y_predicted = self.predict()
        Y = self.Y 
        m = len(Y)
        self.b[0] = self.b[0] - (learning_rate * ((1/m) * np.sum(Y_predicted - Y)))
        self.b[1] = self.b[1] - (learning_rate * ((1/m) * np.sum((Y_predicted - Y) * self.X)))


    def predict(self, X = []) : 
        Y_predicted = np.array([])
        if not X: X = self.X
        b = self.b 
        for x in X : 
            Y_predicted = np.append(Y_predicted, b[0] + (b[1] * x))

        return Y_predicted

    
    def get_current_accuracy(self, Y_predicted) : 
        p, e = Y_predicted, self.Y 
        n = len(Y_predicted)

        return 1 - sum( 
            [
                abs(p[i] - e[i])/e[i] 
                for i in range(n) if e[i] != 0
            ]
        )/n
    

    def compute_cost(self, Y_predicted) : 
        m = len(self.Y)
        J = (1/2*m) * (np.sum(Y_predicted - self.Y)**2)
        return J

    
    def plot_best_fit(self, Y_predicted, fig) : 
        f = plt.figure(fig)
        plt.scatter(self.X, self.Y, color = 'b')
        plt.plot(self.X, Y_predicted, color = 'g')
        f.show()


def main() : 
    X = np.array([i for i in range(11)])
    Y = np.array([2*i for i in range(11)])

    regressor = LinearRegression(X, Y)

    iterations = 0 
    steps = 100 
    learning_rate = 0.01
    costs = []

    # original best fit line
    Y_predicted = regressor.predict()
    regressor.plot_best_fit(Y_predicted, 'Initial best fit line')


    while True : 
        Y_predicted = regressor.predict()
        cost = regressor.compute_cost(Y_predicted)
        costs.append(cost)
        regressor.update_coefficients(learning_rate)

        iterations += 1
        if iterations % steps == 0 : 
            print(iterations, 'epochs elapsed')
            print('Current accuracy is : ', regressor.get_current_accuracy(Y_predicted))

            stop = input("DO you want to stop (Y/N) ? ")
            if stop == 'Y' : 
                break 

        # final best fit line
        regressor.plot_best_fit(Y_predicted, 'final best fit line')

        # plot to verify cost function decrease 
        h = plt.figure('Verification')
        plt.plot(range(iterations), costs, color = 'b')
        h.show()

        # if user wants to predict using regressor
        regressor.predict([i for i in range(10)])

if __name__ == '__main__' : 
    main()