import numpy as np
import pandas as pd
from scipy.io import loadmat

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

def load_ekg(filename: str) -> tuple((np.ndarray, dict[str: None])):
    """
    Loads all 
    """
    # Load the ekg voltage data from the .mat file
    ekg = loadmat(filename + '.mat')['val']

    # Load the features from the .hea file
    f = open(filename + '.hea', 'r')
    features = {}
    for line in f:
        if line[0] == '#':
            contents = line.split(' ')[1:]
            features[contents[0]] = contents[1][:len(contents[1])-1]
    return ekg, features

def get_ekg_features(ekg: np.ndarray) -> pd.DataFrame:
    """
    Takes in an ekg and a list of features and returns them
    """
    pass

def load_ekgs() -> tuple((np.ndarray, dict[str: None])):
    """
    Loads all ekgs in an np.ndarray
    """
    pass

if __name__ == '__main__':
    ekg, features = load_ekg('training/chapman_shaoxing/g1/JS00001')
    print(features)
