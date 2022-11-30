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

def remove_baseline_wander(signal):
    """
    Removes baseline wander from all leads, takes nd-array as input
    """
    proc_signal = np.ndarray((0, signal.shape[1]))
    for x in signal:
        ssds = np.zeros((3))

        cur_lp = np.copy(x)
        iterations = 0
        while True:

            lp, hp = pywt.dwt(cur_lp, "db4")

            ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

            if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
                break

            cur_lp = lp[:]
            iterations += 1

        baseline = cur_lp[:]
        for _ in range(iterations):
            baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")
        new = x - baseline[: len(x)]
        proc_signal = np.vstack((proc_signal, new))
        print(proc_signal.shape, signal.shape)
    return proc_signal

def get_ekg_features(ekg: np.ndarray) -> pd.DataFrame:
    """
    Takes in an ekg and a list of features and returns them
    """
    pass
