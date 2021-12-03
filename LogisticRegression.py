# f(x) = wx + b
# Apply Sigmoid function --> 1/1+e(-wx+b)
# Cost Function --> CrossEntropy
# Optimize Cost function with regard to the Parameter (w,b)
# Using Gradinet Descent to optimize the cost function
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import  matplotlib.pyplot as plt



class LogisticRegression():
    def __init__(self, lr=0.01, n_iters=200):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self,X,y):
        # Init parameters
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        # gradient Descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weight) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum((y_predicted - y))
            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self,X):
        linear_model = np.dot(X, self.weight) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls


    def sigmoid(self,X):
        return  1 / (1 + np.exp(-X))

bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=1234)

def accuracy(y_precicted, y_true):
    y_predicted = np.sum(y_precicted == y_true) / len(y_precicted)
    return y_predicted

regressor = LogisticRegression(lr = 0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
y_predicted = regressor.predict(X_test)

print("Logistic Regression Classification Accuracy = ", accuracy(y_predicted, y_test))
