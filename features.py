from sklearn import preprocessing
from scipy.signal import find_peaks
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