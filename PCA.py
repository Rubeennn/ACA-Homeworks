import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.reducted = None
        self.components_ = None

    def fit(self, X):
        X = X - X.mean(axis=0)

        cov = np.cov(X.T)

        eig_vals, eig_vecs = np.linalg.eigh(cov)

        eig_vals = eig_vals[::-1]
        eig_vecs = eig_vecs[::-1]

        eig_vecs = eig_vecs.T
        needed_comps = eig_vecs[:self.n_components, :]
        self.reducted = np.dot(X, needed_comps.T)
        self.components_ = needed_comps

        return self.reducted
