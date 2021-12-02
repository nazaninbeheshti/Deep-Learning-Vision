import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import  Counter
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def Euclidean_distance(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)

class KNN():
    def __init__(self,k=3):
        self.k = k
    def fit(self,X,y):
        self.X_train = X
        self.Y_train = y

    def predict(self,X):
        predicted_label = [self._predict(x) for x in X]
        return np.array(predicted_label)

    def _predict(self,x):
        #compute the distances
        distances = [Euclidean_distance(x, x_train) for x_train in self.X_train]
        #get K nearest samples and their corresponding labels
        k_indices = np.argsort(distances )[0:self.k]
        #get lables
        K_nearest_labels = [self.Y_train[i] for i in k_indices]
        #most common labels
        most_common = Counter(K_nearest_labels).most_common(1)
        return most_common[0][0]


iris = datasets.load_iris()
X,y = iris.data, iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#print(X_train.shape, X_train.shape)
#print(Y_train.shape)
#plt.figure()
#plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap, edgecolors='k', s=20)
#plt.show()

clf = KNN(k=3)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
acc = np.sum(predictions == Y_test) / len(Y_test)
print(f'Test Accuracy: ', acc)



