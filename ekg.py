import numpy as np
from pandas import DataFrame
from scipy.io import loadmat
from scipy.signal import filtfilt, iirnotch
from features import beat_characteristics
import os
from pywt import dwt, idwt
from scipy.signal import iirnotch, filtfilt
from globals import *
from itertools import groupby


# Dictionary that maps a diagnosis code to the diagnosis name
DIAGNOSES = {
    "164917005": "EKG: Q wave abnormal",
    "251198002": "Clockwise cardiac rotation",
    "428750005": "Nonspecific ST-T abnormality on electrocardiogram",
    "17338001": "Ventricular premature beats",
    "426783006": "ECG: sinus rhythm",
    "427084000": "ECG: sinus tachycardia",
    "251173003": "Atrial bigeminy",
    "698252002": "Non-specific intraventricular conduction delay",
    "164912004": "EKG: P wave abnormal",
    "233917008": "Atrioventricular block",
    "47665007": "Right axis deviation",
    "164937009": "EKG: U wave abnormal",
    "164889003": "ECG: atrial fibrillation",
    "251180001": "Ventricular trigeminy",
    "251146004": "Low QRS voltages",
    "164947007": "Prolonged PR interval",
    "164890007": "EKG: atrial flutter", # THIS ONE RIGHT HERE
    "111975006": "Prolonged QT interval",
    "164934002": "EKG: T wave abnormal",
    "55827005": "Left ventricular hypertrophy",
    "428417006": "Early repolarization",
    "446358003": "Right atrial hypertrophy",
    "59931005": "Inverted T wave",
    "164931005": "ST elevation",
    "59118001": "Right bundle branch block",
    "39732003": "Left axis deviation",
    "164865005": "EKG: myocardial infarction",
    "251199005": "Counterclockwise cardiac rotation",
    "284470004": "Premature atrial contraction",
    "67751000119106": "Right atrial enlargement",
    "270492004": "First degree atrioventricular block",
    "429622005": "ST Depression",
    "164909002": "EKG: left bundle branch block",
    "426177001": "ECG: sinus bradycardia",
    "164873001": "ECG:left ventricle hypertrophy"
}

def load_ekgs() -> tuple((np.ndarray, DataFrame, DataFrame)):
    """
    Load all ekgs from the trainig folder
    """
    # Lists to hold data from each ekg
    ekgs = []
    features = []
    diagnoses = []

    # First set of directories
    sources = os.listdir('./training')
    sources.remove('.DS_Store')
    sources.remove('index.html')

    for source in sources:
        # Second set of directories
        print('getting data from:', source)
        gs = os.listdir('./training/' + source)
        gs.remove('index.html')
        for g in gs:
            # ekg files
            path = './training/' + source + '/' + g
            for file in [x for x in os.listdir(path) if x[-4:] == '.mat']:
                # Loading the file might fail, if so then skip it
                try:
                    ekg, feature, diagnosis, err = load_ekg(path + '/' + file[:-4])
                    ekg = bring_ekg_med_to_zero(ekg)
                    if err:
                        continue
                except:
                    continue
                    
                # If the waveform is less than 5000 points, then skip it
                if ekg.shape[1] < 5000:
                    continue

                # Append data to corresponding lists
                ekgs.append(ekg[:, :5000])
                features.append(feature)
                diagnoses.append(diagnosis)
    
    # Format data into np.ndarrays and pd.DataFrames
    return np.array(ekgs), DataFrame(data=features), DataFrame(data=diagnoses)

def load_ekg(filename: str) -> tuple((np.ndarray, dict[str: None])):
    """
    The data from an ekg
    """
    # Load the ekg voltage data from the .mat file
    ekg = loadmat(filename + '.mat')['val']

    # If too much wander, error
    if too_much_wander(ekg[1]):
        return None, None, None, 'too much wander, ekg discarded'

    # Load the features from the .hea file
    f = open(filename + '.hea', 'r')
    features = {}
    diagnoses = dict.fromkeys(DIAGNOSES.values(), 0)
    for line in f:
        # Any line that starts with # contains data that is useful
        if line[0] == '#':
            # Split by spaces
            contents = line.split(' ')[1:]

            # Get the name and value for the features
            feature_name = contents[0][:-1]
            feature_val = contents[1][:len(contents[1])-1]

            # Skip over any NaNs
            if feature_val == 'NaN':
                continue
            
            # Age is an integer
            if feature_name == 'Age':
                features[feature_name] = int(feature_val)
            
            # Sex is encoded as binary: 1 is male, 0 is female
            if feature_name == 'Sex':
                features[feature_name] = {'Male': 1, 'Female': 0}[feature_val]
            
            # For now, only use atrial fibrillation
            if feature_name == 'Dx' and feature_val == '164889003':
                for dx in feature_val.split(','):
                    dx = dx.strip()
                    diagnoses[DIAGNOSES[dx]] = 1
            
    return ekg, features, diagnoses, None

def too_much_wander(ekg):
    """
    Detects too much wander
    """
    global_max = np.max(ekg)
    peaks = ekg > global_max * (1 - PEAK_DRIFT_THRESHOLD)
    if sum([k for k, _ in groupby(peaks)]) < NUM_QRS_PEAKS_THRESHOLD:
        return True
    return False

def remove_baseline_wander(ekg, fs=500):
    """
    Removes baseline wandering from each lead
    """
    for i, e in enumerate(ekg):
        b, a = iirnotch(0.05 , Q=0.005, fs=fs)
        filtered_ecg = filtfilt(b, a, e)
        median = np.median(filtered_ecg)
        filtered_ecg = filtered_ecg-median
        ekg[i] = filtered_ecg

def is_invalid(ekg):
    for lead_num in range(12):
        HR, RR_var, RR_var_normalized = beat_characteristics(ekg,lead_num)
        if RR_var > 100000:
            return True
    return False

def bring_ekg_med_to_zero(ekg):
    """
    bring the median of a single lead to zero
    """
    return ekg - np.median(ekg)

def downsample(ekgs, num_points=100, lead_index=1):
    down_sampled_ekgs = []
    for ekg in ekgs:
        down_sampled_lead = _downsample(ekg[lead_index])
        down_sampled_ekgs.append(down_sampled_lead)
    down_sampled_ekgs = np.array(down_sampled_ekgs) / 1000
    return down_sampled_ekgs

def _downsample(ekg, num_points=100):
    new_ekg = []
    sample_window = ekg.shape[0] // num_points
    goal_step_size = (ekg.shape[0]-sample_window) / num_points
    extrema_window_size = 5 * sample_window
    window_lefts = [0]
    current_sum = 0
    for i in range(1, num_points):
        if current_sum / i < goal_step_size:
            window_lefts.append(window_lefts[-1] + sample_window+1)
            current_sum += sample_window+1
        else:
            window_lefts.append(window_lefts[-1] + sample_window)
            current_sum += sample_window
        
    for i in window_lefts:
        left_extrema_window = max(0, i-extrema_window_size//2)
        right_extrema_window = min(i+extrema_window_size//2, ekg.shape[0]-1)
        extrema_window = ekg[left_extrema_window:right_extrema_window+1]
        max_extrema_window = np.argmax(extrema_window) + left_extrema_window
        min_extrema_window = np.argmin(extrema_window) + left_extrema_window
        if i <= max_extrema_window < i+sample_window:
            new_ekg.append(ekg[i])
        elif i <= min_extrema_window < i+sample_window:
            new_ekg.append(ekg[i])
        else:
            new_ekg.append(np.average(ekg[i:i+sample_window]))
    return np.array(new_ekg)

def merge_ekgs_features(ekgs, features_array, features_tree):
    down_sampled_ekgs_with_features = []
    for i in features_tree.index.values.tolist():
        row = np.zeros((2 + ekgs.shape[1] + features_array.shape[1], ))
        row[2:2+features_array.shape[1]] = features_array[i]
        row[-ekgs.shape[1]:] = ekgs[i]
        down_sampled_ekgs_with_features.append(row)
    down_sampled_ekgs_with_features = np.array(down_sampled_ekgs_with_features)
    down_sampled_ekgs_with_features[:, :2] = features_tree[['Age', 'Sex']].to_numpy()
    return down_sampled_ekgs_with_features

def one_interval(ekg, downsample=True):
    global_max = np.max(ekg)
    
    # Find first QRS peak
    i = 0
    bucket = []
    while ekg[i] < global_max * (1 - PEAK_DRIFT_THRESHOLD):
        i += 1
    while ekg[i] > global_max * (1 - PEAK_DRIFT_THRESHOLD):
        bucket.append((ekg[i], i))
        i += 1
    first_peak = max(bucket)[1]

    # Find second QRS peak
    i += 10
    bucket = []
    while ekg[i] < global_max * (1 - PEAK_DRIFT_THRESHOLD):
        i += 1
    while ekg[i] > global_max * (1 - PEAK_DRIFT_THRESHOLD):
        bucket.append((ekg[i], i))
        i += 1
    second_peak = max(bucket)[1]

    # Distance between the two peaks
    window_size = second_peak - first_peak

    # Go to the bottom of the first QRS peak
    look_ahead = 3
    i = first_peak
    while ekg[i] > ekg[i-look_ahead]:
        i -= 1
    
    # Go to the baseline
    baseline = np.median(ekg)
    while ekg[i] < baseline:
        i -= 1
    
    # get the minimum value between i and i+look_ahead
    return ekg[i:i+window_size+1]
