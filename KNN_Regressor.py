import numpy as np


class KNN:
    def __init__(self, n_k):
        self.n_k = n_k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    @staticmethod
    def distance(x1, x2):
        return np.linalg.norm(x1 - x2, axis=1)

    def predict(self, X):
        pred = []

        # In case if we do regression for one data point
        if len(X) == 1 or len(X.shape) == 1:
            distance = self.distance(X, self.X_train)
            indices = np.argsort(distance)[:self.n_k]
            pred.append(np.mean(self.y_train[indices]))
        else:
            for x in X:
                distance = self.distance(x, self.X_train)
                indices = np.argsort(distance)[:self.n_k]
                pred.append(np.mean(self.y_train[indices]))

        return pred

# Can check the regressor here.
# a = np.array([2, 5, 1.5])
# b = np.array([[5, 2, 4], [3, 2, 0], [5, 0, 3], [4, 5, 8], [4, 1, 9], [2, 5, 0]])
# y = np.array([5, 3, 6, 5, 2, 3])
#
# q = KNN(n_k=4)
#
# q.fit(b, y)
# prediction = q.predict(a)
# print(prediction)
