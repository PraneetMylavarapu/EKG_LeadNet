from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from ekg import get_ekg_features, load_ekgs
from trees import baseline_tree
# from networks import baseline_network
from random import sample

ekgs, features, diagnoses = load_ekgs()
features = features.join(diagnoses['ECG: atrial fibrillation'])

# Get equal amount of label=1 and label=0
print('Balancing data...')
features1 = features[features['ECG: atrial fibrillation'] == 1]
n = len(features1)
features0 = features[features['ECG: atrial fibrillation'] == 0]
features0 = features0.sample(n=n)

# Get corresponding ekg waveforms
ekgs_temp = []
for i in features0.index.values.tolist():
    ekgs_temp.append(ekgs[i])
for i in features1.index.values.tolist():
    ekgs_temp.append(ekgs[i])

# Finalize variables
ekgs = ekgs_temp
features = pd.concat([features0, features1])
features = features.reset_index()

# Extract features from waveforms
print('Extracting features...')
features2 = []
ekgs_temp = []
for ekg in ekgs:
    fs = get_ekg_features(ekg)
    
    # Add the features to the decision tree data
    features2.append(fs)

    # Insert the features to the ekg of the ekg array so they can be
    # used to train the neural network
    f = np.array(list(fs.values()))
    ekg_temp = np.zeros((ekg.shape[0], ekg.shape[1]+len(f)))
    ekg_temp[:, :ekg.shape[1]] = ekg
    ekg_temp[:, -len(f):] = f
    ekgs_temp.append(ekg_temp)
ekgs = np.array(ekgs_temp)

features2 = pd.DataFrame(data=features2)
features_tree = features.join(features2)


print('training decision tree...')
baseline_tree(features_tree, 'ECG: atrial fibrillation')
# print('training neural network...')
# baseline_network(features, ekgs, 'ECG: atrial fibrillation')