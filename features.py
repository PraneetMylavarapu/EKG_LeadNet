from sklearn import preprocessing
from scipy.signal import find_peaks
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

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.

    return data[s<m]

def get_R_indices(ekg, lead_num=2, R_factor=5, distance=100):
    mad = median_abs_deviation(ekg[lead_num])
    R_indices, _ = find_peaks(ekg[lead_num], height=R_factor*mad, distance=distance)

    return R_indices

def get_avg_RR_interval(ekg, lead_num=2, R_factor=5, distance=100):
    R_indices = get_R_indices(ekg, lead_num, R_factor, distance)
    rr_intervals = []
    for i in range(len(R_indices)-1):
        rr_intervals.append(R_indices[i+1] - R_indices[i])
    rr_intervals = np.array(rr_intervals)
    rr_intervals = reject_outliers(rr_intervals, 3)
    avg_rr_interval = np.mean(rr_intervals)

    return avg_rr_interval

def get_QS_indices(ekg, lead_num=2, window_factor=4, R_factor=5, distance=100):
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

def get_QRS_interval(ekg, lead_num=2, window_factor=4, R_factor=5, distance=100):

    Q_indices, S_indices = get_QS_indices(ekg, lead_num, window_factor, R_factor, distance)
    # average QRS interval

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
    qrs_intervals = reject_outliers(qrs_intervals, 3)
    avg_qrs_interval = np.mean(qrs_intervals)

    QRS_var = statistics.variance(qrs_intervals)

    return avg_qrs_interval, QRS_var



