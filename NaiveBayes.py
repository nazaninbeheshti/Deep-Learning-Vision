# P(y|X) = P(X|y) . P(y) / P(X)
# X = (x1, x2, x3, ... , xn) is a feature Vector --> all feature are mutually independent
# P(y|X) = P(x1|y) . P(x2|y) ...... . p(xn|y). p(y) / P(X)
# Select class with highest probablity
# y = argmax(y) P(x1|y) . P(x2|y) ...... . p(xn|y). p(y) --> Very small numbers --> overflow
# y = argmax(y) log( P(x1|y) ) + log (P(x2|y)) + ..... + log (P(xn|y)) + log(p(y))
# p(xi|y) -->  class Guassian distribution

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class NaiveBayes():


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        #init mean and variance and priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[c == y]
            self._mean[idx,:] = X_c.mean(axis=0)
            self._var[idx,: ] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples) # Frequency of class in training samples

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # Calculate posterior probablity for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            #class_conditional = np.sum(self._pdf(idx, c))
            posterior = np.sum(np.log(self._pdf(idx, x)))
            #posterior = prior * class_conditional
            posterior = prior + posterior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x-mean)**2) / (2*var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator/denominator



def accuracy(y_pred, y_true):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=123)
model = NaiveBayes()
model.fit(X_train, Y_train)
predections = model.predict(X_test)
print(len(predections))
print(Y_test.shape)
print("Naive Bias Classification Accuracy = ", accuracy(predections, Y_test))



