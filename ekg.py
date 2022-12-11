import numpy as np
from pandas import DataFrame
from scipy.io import loadmat
from scipy.signal import filtfilt, iirnotch
from features import beat_characteristics
import os
from pywt import dwt, idwt
from scipy.signal import iirnotch, filtfilt


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
    "164890007": "EKG: atrial flutter",
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
                    ekg, feature, diagnosis = load_ekg(path + '/' + file[:-4])
                    # remove_baseline_wander(ekg)
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


def load_ekgs_fast() -> tuple((np.ndarray, DataFrame, DataFrame)):
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
                    ekg, feature, diagnosis = load_ekg(path + '/' + file[:-4])
                    # remove_baseline_wander(ekg)
                except:
                    continue
                    
                # If the waveform is less than 5000 points, then skip it
                if ekg.shape[1] < 5000:
                    continue

                # Append data to corresponding lists
                ekgs.append(ekg[:, :5000])
                features.append(feature)
                diagnoses.append(diagnosis)
            break
        break
    
    # Format data into np.ndarrays and pd.DataFrames
    return np.array(ekgs), DataFrame(data=features), DataFrame(data=diagnoses)

def load_ekg(filename: str) -> tuple((np.ndarray, dict[str: None])):
    """
    The data from an ekg
    """
    # Load the ekg voltage data from the .mat file
    ekg = loadmat(filename + '.mat')['val']

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
            
    return ekg, features, diagnoses

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

def remove_baseline_wander_old(signal):
    """
    Removes baseline wander from all leads, takes nd-array as input
    """
    proc_signal = np.ndarray((0, signal.shape[1]))
    for x in signal:
        ssds = np.zeros((3))

        cur_lp = np.copy(x)
        iterations = 0
        while True:
            # Decompose 1 level
            lp, hp = dwt(cur_lp, "db4")

            # Shift and calculate the energy of detail/high pass coefficient
            ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

            # Check if we are in the local minimum of energy function of high-pass signal
            if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
                break

            cur_lp = lp[:]
            iterations += 1

        # Reconstruct the baseline from this level low pass signal up to the original length
        baseline = cur_lp[:]
        for _ in range(iterations):
            baseline = idwt(baseline, np.zeros((len(baseline))), "db4")
        new = x - baseline[: len(x)]
        proc_signal = np.vstack((proc_signal, new))
    return proc_signal

def is_invalid(ekg):
    for lead_num in range(12):
        HR, RR_var, RR_var_normalized = beat_characteristics(ekg,lead_num)
        if RR_var > 100000:
            return True
    return False

def remove_baseline_wander1(ekg, fs, cutoff=0.05, Q=0.005):
    """
    remove baseline wander from a single lead
    """
    b, a = iirnotch(cutoff , Q = Q, fs = fs)
    ekg = filtfilt(b, a, ekg)
    return ekg

def bring_ekg_med_to_zero(ekg):
    """
    bring the median of a single lead to zero
    """
    ekg = ekg - np.median(ekg)
    return ekg
