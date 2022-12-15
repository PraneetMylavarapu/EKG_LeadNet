from sklearn import preprocessing
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
import statistics
import numpy as np
import pandas as pd
from globals import *

def get_features(ekg: np.ndarray, leads=[i for i in range(12)]) -> pd.DataFrame:
    """
    Takes in an ekg and a list of features and returns them
    """
    features = {}
    HR, RR_var, RR_var_normalized = beat_characteristics(leads, lead_num=1)
    QRS_interval = None
    QRS_area = None
    DAVID_interval = None
    DAVID_area = None
    ekg_features = {}
    try:
        # avg_QRS_interval = get_avg_qrs_interval(ekg)
        ekg_features = get_QRS_interval(ekg[1])
    except:
        print('QRS failed')
        QRS_interval, QRS_area = get_QRS_interval(ekg[1])
        QRS_interval = np.nan
        QRS_area = np.nan
        DAVID_interval = np.nan
        DAVID_area = np.nan

    # Features that are independent of lead

    features['HR'] = HR
    features['RR_var'] = RR_var
    for key in ekg_features:
        features[key] = ekg_features[key]

    # Features that are calculated per lead
    for i in leads:
        lead = ekg[i]
        features['difference' + str(i)] = max_peak_height(lead) * 0.001
        features['area' + str(i)] = lead_area(lead) * 0.001
        
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

def reject_outliers(data:np.ndarray, m:float = 3.):
    """
    Filters statistically extraordinary intervals between peaks 

    params:
    data: np.ndarray of intervals (QRS, RR, etc)
    m: number of mdevs to keep

    returns:
    data: np.ndarray of statistically relevant intervals
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.

    return data[s<m]

def get_R_indices(ekg:np.ndarray, lead_num:int=1, R_factor:float=5., distance:int=100):
    """
    Gets the indices of R peaks in a single lead 
    
    params:
    ekg: np.ndarray of a single lead
    lead_num: lead number
    R_factor: multiplier to the mdev of ekg to provide minimum height of the R peak
    distance: minimum distance between peaks

    returns:
    R_indices: np.ndarray of indices of R peaks
    """
    mad = median_abs_deviation(ekg[lead_num])
    R_indices, _ = find_peaks(ekg[lead_num], height=R_factor*mad, distance=distance)

    return R_indices

def get_avg_RR_interval(ekg:np.ndarray, lead_num:int=1, R_factor:float=5., distance:int=100, significane:float=3.):
    """
    Gets the average RR interval in a single lead
    
    params:
    ekg: np.ndarray of a single lead
    lead_num: lead number
    R_factor: multiplier to the mdev of ekg to provide minimum height of the R peak
    distance: minimum distance between peaks
    significane: number of mdevs to keep

    returns:
    avg_rr_interval: average RR interval
    """
    R_indices = get_R_indices(ekg, lead_num, R_factor, distance)
    rr_intervals = []
    for i in range(len(R_indices)-1):
        rr_intervals.append(R_indices[i+1] - R_indices[i])
    rr_intervals = np.array(rr_intervals)
    rr_intervals = reject_outliers(rr_intervals, significane)
    avg_rr_interval = np.mean(rr_intervals)

    return avg_rr_interval

def get_QS_indices(ekg:np.ndarray, lead_num:int=1, window_factor:float=4., R_factor:float=5., distance:int=100):
    """
    Gets the indices of Q and S peaks in a single lead 
    
    params:
    ekg: np.ndarray of a single lead
    lead_num: lead number
    window_factor: how much % of the average RR interval to look around the R peak
    R_factor: multiplier to the mdev of ekg to provide minimum height of the R peak
    distance: minimum distance between peaks

    returns:
    Q_indices: np.ndarray of indices of Q peaks
    S_indices: np.ndarray of indices of Q peaks
    """
    R_indices = get_R_indices(ekg, lead_num, R_factor, distance)
    avg_rr_interval = get_avg_RR_interval(ekg, lead_num, R_factor, distance)
    S_indices = []
    Q_indices = []
    for i in range(1, len(R_indices)-1):
        window = int(avg_rr_interval/window_factor)
        lhs = ekg[lead_num][R_indices[i]-window:R_indices[i]]
        rhs = ekg[lead_num][R_indices[i]:R_indices[i]+window]
        potential_Q_indices, _ = find_peaks(-lhs)
        Q_index = R_indices[i]-window+potential_Q_indices[np.argmax(-lhs[potential_Q_indices])]
        Q_indices.append(Q_index)
        potential_S_indices, _ = find_peaks(-rhs)
        S_index = R_indices[i]+potential_S_indices[np.argmax(-rhs[potential_S_indices])]
        S_indices.append(S_index)

    return Q_indices, S_indices

def get_avg_qrs_interval(ekg:np.ndarray, lead_num:int=1, window_factor:float=4., R_factor:float=5., distance:int=100, significance:float=3.):
    """
    Gets the average QRS interval in a single lead 
    
    params:
    ekg: np.ndarray of a single lead
    lead_num: lead number
    window_factor: how much % of the average RR interval to look around the R peak
    R_factor: multiplier to the mdev of ekg to provide minimum height of the R peak
    distance: minimum distance between peaks
    significane: number of mdevs to keep

    returns:
    avg_qrs_interval: average QRS interval
    """
    Q_indices, S_indices = get_QS_indices(ekg, lead_num, window_factor, R_factor, distance)

    # find first zero crossing before Q index
    Q_zero_crossings = []
    for Q_index in Q_indices:
        i = Q_index
        flag = 1
        while ekg[lead_num][i] < 0:
            i = i - 1
            if i < 0:
                flag = 0
                break
        if flag:
            Q_zero_crossings.append(i)
    Q_zero_crossings = np.array(Q_zero_crossings)

    # find first zero crossing after S index
    S_zero_crossings = []
    for S_index in S_indices:
        i = S_index
        flag = 1
        while ekg[lead_num][i] < 0:
            i = i + 1
            if i > len(ekg[lead_num])-2:
                flag = 0
                break
        if flag:
            S_zero_crossings.append(i)
    S_zero_crossings = np.array(S_zero_crossings)

    qrs_intervals = S_zero_crossings - Q_zero_crossings
    qrs_intervals = reject_outliers(qrs_intervals, significance)
    avg_qrs_interval = np.mean(qrs_intervals)

    # QRS_var = statistics.variance(qrs_intervals)

    return avg_qrs_interval


def get_QRS_interval(ekg):
    features = {}
    global_max = np.max(ekg)
    look_ahead = 3
    
    # Find first QRS peak
    i = 0
    bucket = []
    while ekg[i] < global_max * (1 - PEAK_DRIFT_THRESHOLD):
        i += 1
    while ekg[i] >= global_max * (1 - PEAK_DRIFT_THRESHOLD):
        bucket.append((ekg[i], i))
        i += 1
    first_peak = max(bucket)[1]

    # Go to the bottom of the first QRS peak
    i = first_peak
    while ekg[i] > ekg[i-look_ahead]:
        i -= 1
    
    # Go to the baseline
    while ekg[i-look_ahead] > ekg[i]:
        i -= 1
    
    # Left endpoint
    left = max(0, i)

    # Go to the bottom of the first QRS peak
    i = first_peak
    while ekg[i] > ekg[i+look_ahead]:
        i += 1
    
    # Go to the baseline
    while ekg[i+look_ahead] > ekg[i]:
        i += 1
    
    # Right endpoint
    right = i

    # QRS interval in ms
    features['QRS_interval'] = (right - left) * 0.001
    
    # QRS area under curve
    features['QRS_area'] = np.trapz(ekg[left:right] - np.min(ekg[left:right])) * 0.001

    # Go to 2nd QRS peak
    while ekg[i] < global_max * (1 - PEAK_DRIFT_THRESHOLD) and i < ekg.shape[0]-look_ahead-1:
        i += 1
    
    # Go to bottom of 2nd QRS peak
    while ekg[i] > ekg[i-look_ahead] and i != ekg.shape[0]-look_ahead-2:
        i -= 1
    
    # Look at a new window
    window_size = max((i - left) * 0.001, 1) # If i == left then set it to 1
    left = right
    if left >= i:
        right = left
    else:
        right = np.argmax(ekg[left:i]) + left

    # DAVID interval in ms
    features['DAVID_interval'] = (right - left) * 0.001
    
    # DAVID area under curve
    features['DAVID_area'] = 0
    if features['DAVID_interval'] != 0:
        features['DAVID_area'] = np.trapz(ekg[left:right] - np.min(ekg[left:right])) * 0.001

    # DAVID peak ratio
    features['DAVID_peak_ratio'] = ekg[right] / global_max
    features['DAVID_window_ratio'] = features['DAVID_interval']  / window_size
    
    return features

