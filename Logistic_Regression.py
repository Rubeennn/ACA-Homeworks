import numpy as np


class log_reg:
    def __init__(self, iterations, learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate
        pass

    @staticmethod
    def sigmoid(X):
        sigmoid = 1 / (1 + np.exp(-X))

        return sigmoid

    def init_weights(self, X):
        self.W = np.r_[np.ones((1, 1)), np.random.rand(X.shape[1], 1)]

        return self.W

    def fit(self, X, y):
        X = np.insert(X, 0, 1, 1)
        for i in range(self.iterations):
            lin = np.dot(X, self.W)
            pred = self.sigmoid(lin)
            dw = (1 / X.shape[0]) * np.dot(X.T, (pred - y))
            self.W -= dw * self.learning_rate

    def pred(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        pred = np.dot(X, self.W)
        pred = self.sigmoid(pred)
        pred = (pred >= 0.5).astype(int)
        return pred




