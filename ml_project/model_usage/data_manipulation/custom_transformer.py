import numpy as np
from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin

np.random.seed(42)


class RandomCustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed = randint(0, 10) * X_transformed / randint(10, 100)
        return X_transformed
