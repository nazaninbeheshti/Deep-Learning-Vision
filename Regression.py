import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import  matplotlib.pyplot as plt

class BaseRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_featues = X.shape

        #init parameters
        self.weight = np.zeros(n_featues)
        self.bias = 0

        #gradinet descent
        for _ in range(self.n_iters):
            y_predicted = self._approximation(X, self.weight, self.bias)

            #Computee gradients
            dw = (1/n_samples) * np.dot(X.T , (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            #Update Parameters
            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self,X):
        return self._predict(X, self.weight, self.bias)

    def _predict(self,X, w, b):
        raise NotImplementedError()

    def _approximation(self, X, w, b):
        raise NotImplementedError()




class LinearRegression(BaseRegression):


    def _approximation(self, X, w, b):
        return np.dot(X,w) + b

    def _predict(self, X, w, b):
        return  np.dot (X, w) + b

class LogisticRegression(BaseRegression):


    def _approximation(self, X, w, b):
        linear_model = np.dot(X,w) + b
        return self._sigmoid(linear_model)

    def _sigmoid(self,x):
        return 1 / (1+np.exp(-x))

    def _predict(self,X, w, b):
        linear_model = np.dot(X, w) + b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return  y_predicted_cls


if __name__ == "__main__":

    def accuracy(y_precicted, y_true):
        y_predicted = np.sum(y_precicted == y_true) / len(y_precicted)
        return y_predicted

    def squared_mean_error(y_predicted, y_true):
        return np.sum((y_predicted - y_true)**2)

    def r2_Score(y_predicted, y_true):
        corr_matrix = np.corrcoef(y_true, y_predicted)
        corr = corr_matrix[0,1]
        return corr ** 2

    # Test Logistic Regression
    bc = datasets.load_breast_cancer()
    X,y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=1234)
    regressor = LogisticRegression(lr = 0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    y_predicted = regressor.predict(X_test)
    print("Logistic Regression Classification Accuracy = ", accuracy(y_predicted, y_test))

    # Test Linear Regression
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    y_predicted = regressor.predict(X_test)
    accu = r2_Score(y_predicted, y_test)
    print("Linear Regression accuracy = ", accu)

