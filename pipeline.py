import pandas as pd
import numpy as np
from ekg import load_ekgs, load_ekgs_fast
from features import get_features
from trees import baseline_tree, short_tree, log_loss_tree, entropy_tree, big_forest, big_forest_small_tree
from globals import *
# from networks import baseline_network, cnn

ekgs, features, diagnoses = load_ekgs()
# ekgs, features, diagnoses = load_ekgs_fast()
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
ekgs = np.array(ekgs_temp)
features = pd.concat([features0, features1])
features = features.reset_index()

# Extract features from waveforms
print('Extracting features...')
features2 = []
ekgs_temp = []
for ekg in ekgs:
    fs = get_features(ekg)
    
    # Add the features to the decision tree data
    features2.append(fs)

    # Insert the features to the ekg of the ekg array so they can be
    # used to train the neural network
    f = np.array(list(fs.values()))
    ekg_temp = np.zeros((ekg.shape[0], ekg.shape[1]+len(f)))
    ekg_temp[:, :ekg.shape[1]] = ekg
    ekg_temp[:, -len(f):] = f
    ekgs_temp.append(ekg_temp)
ekgs_with_features = np.array(ekgs_temp)

# Replace NaNs from the dataframe array with mean
features2 = pd.DataFrame(data=features2)
features_tree = features.join(features2)
features_tree = features_tree.dropna()
# for col in features_tree.columns:
    # features_tree[col] = features_tree[col].fillna(features_tree[col].median())

# Replace NaNs from the numpy array with mean
for i in range(features_tree.shape[1]):
    ekgs_with_features[:, :, -i] = np.nan_to_num(ekgs_with_features[:, :, -i], np.nanmean(ekgs_with_features[:, :, -i]))
    ekgs_with_features[:, :, -i] = np.max(ekgs_with_features[:, :, -i].flatten())

print('\nrows of data:', features_tree.shape[0])

# Train different models
print('\ntraining baseline decision tree...')
baseline_tree(features_tree, 'ECG: atrial fibrillation', num_iterations=10)
print('\ntraining short decision tree...')
short_tree(features_tree, 'ECG: atrial fibrillation', num_iterations=10)
print('\ntraining big forest...')
big_forest(features_tree, 'ECG: atrial fibrillation', num_iterations=10)
print('\ntraining big forest small tree...')
big_forest_small_tree(features_tree, 'ECG: atrial fibrillation', num_iterations=10)
print('\ntraining entropy decision tree...')
entropy_tree(features_tree, 'ECG: atrial fibrillation', num_iterations=10)
print('\ntraining log_loss decision tree...')
log_loss_tree(features_tree, 'ECG: atrial fibrillation', num_iterations=10)
# print('training baseline neural network...')
# baseline_network(features, ekgs_with_features, 'ECG: atrial fibrillation', lr=0.0001)
# print('training cnn neural network')
# cnn(features, ekgs[:, 1:2, :], 'ECG: atrial fibrillation', lr=1e-6)
