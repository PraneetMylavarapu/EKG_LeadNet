import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

import scipy as sp
from scipy import signal, fftpack
# from scipy.signal import find_peaks, gaussian_filter1d, argrelmax
from scipy.io import loadmat
from statistics import median, mean

import glob
import os
import ntpath

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

def load_ekgs() -> tuple((np.ndarray, dict[str: None])):
    ekgs = []
    features = []
    sources = os.listdir('./training')
    sources.remove('.DS_Store')
    sources.remove('index.html')
    for source in [sources[0]]:
    # for source in sources:
        gs = os.listdir('./training/' + source)
        gs.remove('index.html')
        for g in [gs[0]]:
        # for g in gs:
            path = './training/' + source + '/' + g
            for file in os.listdir(path):
                if file[-4:] == '.mat':
                    print(file)
                    ekg, feature = load_ekg(path + '/' + file[:-4])
                    ekgs.append(ekg)
                    features.append(feature)
    return np.array(ekgs), pd.DataFrame(data=features)


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

if __name__ == '__main__':
    ekg, features = load_ekg('training/chapman_shaoxing/g1/JS00001')
    print(features)

    
def remove_noise(ekg: np.ndarray):
    """
    Removes noise from an ekg by...
    """
    pass


def get_peak_indices(ekg: np.ndarray) -> list[int]:
    """
    Finds peaks in the ekg
    returns:
        peaks: a list of indices where peaks occur in the waveform
    """
    peaks = []
    return peaks
