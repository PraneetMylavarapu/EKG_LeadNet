import numpy as np
import pandas as pd

def get_ekg_features_test() -> pd.DataFrame:
    from sklearn import datasets

    iris = datasets.load_iris()
    data = pd.DataFrame({
        'sepal length': iris.data[:, 0],
        'sepal width':iris.data[:, 1],
        'petal length':iris.data[:, 2],
        'petal width':iris.data[:, 3],
        'species':iris.target
    })

    return data


def get_ekg_features(ekg: np.ndarray, features: list[str]) -> pd.DataFrame:
    """
    Takes in an ekg and a list of features and returns them
    """
    pass

def load_ekgs() -> tuple(np.ndarray, dict[str: _]):
    """
    Loads all ekgs in an np.ndarray
    """
    pass