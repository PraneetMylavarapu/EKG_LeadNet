from sklearn import preprocessing
from scipy.signal import find_peaks,filtfilt, iirnotch
from scipy.stats import median_abs_deviation
import statistics
import numpy as np
import pandas as pd


def get_features(ekg: np.ndarray, leads=[i for i in range(12)]) -> pd.DataFrame:
    """
    Takes in an ekg and a list of features and returns them
    """
    features = {}
    HR, RR_var, RR_var_normalized = beat_characteristics(leads, lead_num=1)

    # Features that are independent of lead
    features['HR'] = HR
    features['RR_var'] = RR_var

    # Features that are calculated per lead
    for i in leads:
        lead = ekg[i]
        features['difference' + str(i)] = max_peak_height(lead)
        features['area' + str(i)] = lead_area(lead)
        
    return features

"""
----------------------------------------------------------------------------
"""

def max_peak_height(lead):
    """
    Calculates the highest point - lowest point
    """
    return max(lead) - min(lead) / (1e3)

def lead_area(lead):
    """
    Calculates the total area under curve of the ekg
    """
    return np.trapz(lead - min(lead)) / (1e6)

def beat_characteristics(ekg, lead_num=1):
    """
    RR_var: variance in distance between beats (distance between peaks)
    HR: Heart Rate
    """
    RR_var = 100
    RR_var_normalized = 100
    HR = 60
    try:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    
        lead = ekg[lead_num][1000:4000]    
        all_peaks, _ = find_peaks(lead, height=max(lead)/1.4, distance=150)

        intervals = np.diff(all_peaks)
        intervals_normalized = scaler.fit_transform(np.reshape(intervals,(-1,1)))
        RR_var = statistics.variance(intervals)
        RR_var_normalized = statistics.variance(np.reshape(intervals_normalized,(1,len(intervals_normalized)))[0])
        HR = 30000/statistics.mean(intervals)
    except:
        pass

    return HR, RR_var, RR_var_normalized

def filter_peaks(peaks:np.ndarray, peak_indices:np.ndarray, window:int, inverted:bool=False) -> np.ndarray:
    if inverted:
        peaks = -1*peaks
    true_peak_indices = []
    i = 0
    while i<len(peak_indices):
        cluster = [[peaks[i], peak_indices[i]]]
        if i<len(peak_indices)-1:
            while peak_indices[i+1]-peak_indices[i] < window:
                cluster.append([peaks[i+1], peak_indices[i+1]])
                i = i+1
                if i == len(peak_indices)-1:
                    break
        true_peak_indices.append(max(cluster)[1])
        i = i+1
    true_peak_indices = np.array(true_peak_indices)
    return true_peak_indices

def _find_peaks(ecg:np.ndarray, min_height:float, max_height:float=None, inverted:bool=False) -> np.ndarray:
    sign = 1
    if inverted:
        sign = -1
    if max_height:
        height = (sign*min_height, sign*max_height)
    else:
        height = sign*min_height
    peak_indices, _ = find_peaks(sign*ecg, height=height)
    return peak_indices

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def get_QRS(ekg, factor=5, margin=100, fs=500):
    b, a = iirnotch(0.05 , Q=0.005, fs=fs)
    bw_fixed_ecg = filtfilt(b, a, ekg[1])
    median = np.median(bw_fixed_ecg)
    bw_fixed_ecg = bw_fixed_ecg-median

    # median absolute deviation (kind of like standard deviation but from the median rather than the mean)
    mad = median_abs_deviation(bw_fixed_ecg)

    # find Rs
    R_indices, _ = find_peaks(bw_fixed_ecg, height=factor*mad)
    Rs = bw_fixed_ecg[R_indices]

    # fitler Rs
    window = 20
    true_R_indices = []
    i = 0
    while i<len(R_indices):
        cluster = [[Rs[i], R_indices[i]]]
        if i<len(R_indices)-1:
            while R_indices[i+1]-R_indices[i] < window:
                cluster.append([Rs[i+1], R_indices[i+1]])
                i = i+1
                if i == len(R_indices)-1:
                    break
        true_R_indices.append(max(cluster)[1])
        i = i+1
    true_R_indices = np.array(true_R_indices)

    # deciding the cutoff for S
    S_factor = 4
    S_cutoff = -S_factor*mad

    cut_ecg = bw_fixed_ecg[true_R_indices[1]+margin:true_R_indices[-2]-margin]

    # finding and filtering Ss
    S_indices = _find_peaks(cut_ecg, min_height=S_cutoff, inverted=True)
    Ss = cut_ecg[S_indices]
    true_S_indices = filter_peaks(Ss, S_indices, window=100, inverted=True)

    # deciding the cutoff for Q
    Q_factor = 4
    Q_cutoff = -Q_factor*mad

    # finding and filtering Qs
    Q_indices = _find_peaks(cut_ecg, min_height=Q_cutoff, max_height=S_cutoff, inverted=True)
    Qs = cut_ecg[Q_indices]
    true_Q_indices = filter_peaks(Qs, Q_indices, window=50, inverted=True)

    # average QRS interval

    # find first zero crossing before Q index
    Q_zero_crossings = []
    for Q_index in true_Q_indices:
        i = Q_index
        flag = 1
        while cut_ecg[i] < 0:
            i = i - 1
            if i < 0:
                flag = 0
                break
        if flag:
            Q_zero_crossings.append(i)
    Q_zero_crossings = np.array(Q_zero_crossings)

    # find first zero crossing after S index
    S_zero_crossings = []
    for S_index in true_S_indices:
        i = S_index
        flag = 1
        while cut_ecg[i] < 0:
            i = i + 1
            if i > len(cut_ecg)-2:
                flag = 0
                break
        if flag:
            S_zero_crossings.append(i)
    S_zero_crossings = np.array(S_zero_crossings)

    # average QRS interval
    qrs_intervals = S_zero_crossings - Q_zero_crossings
    qrs_intervals = reject_outliers(qrs_intervals, 3)
    avg_qrs_interval = np.mean(qrs_intervals)