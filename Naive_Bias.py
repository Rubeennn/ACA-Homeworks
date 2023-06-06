import numpy as np

class Naive:

    def fit(self, X, y):
        self.classes = np.unique(y, return_counts=True)[1]
        self.classes = self.classes.astype(float)
        self.classes_probas = self.classes / np.sum(self.classes)
        self.classes = len(np.unique(y, return_counts=True)[0])

        self.mean = np.zeros((self.classes, X.shape[1]))
        self.std = np.zeros((self.classes, X.shape[1]))

        for i in range(self.classes):
            k = X[np.where(y == i)]
            self.mean[i] = np.mean(k, axis=0)
            self.std[i] = np.std(k, axis=0)

        return self

    def pred(self, X):
        res = []
        for i in range(X.shape[0]):
            probas = []
            for j in range((self.classes)):
                possibilities_for_each_feature = (1 / (np.sqrt(2 * np.pi) * self.std[j])) * np.exp(
                    (-1 * (X[i] - self.mean[j]) ** 2) / (2 * self.std[j] ** 2))
                probas.append(possibilities_for_each_feature.prod() * self.classes_probas[j])

            probas = np.array(probas)
            res.append(probas / probas.sum())
            final_result = np.array(res)

        return np.argmax(final_result, axis=1)


from sklearn.datasets import make_classification

X, y = make_classification(n_samples=50, n_features=8, n_classes=4, n_informative=3, random_state=42)
a = Naive()
a.fit(X, y)
print(a.pred(X))
