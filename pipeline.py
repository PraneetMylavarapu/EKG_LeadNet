import pandas as pd
import numpy as np
from ekg import load_ekgs, downsample
from features import get_features
from trees import baseline_tree, short_tree, log_loss_tree, entropy_tree, big_forest, big_forest_small_tree
from globals import *
from networks import baseline_feature_network, cnn

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
ekgs = np.array(ekgs_temp)
features = pd.concat([features0, features1])
features = features.reset_index()

# Extract features from waveforms
print('Extracting features...')
features2 = []
features_temp = []
for ekg in ekgs:
    fs = get_features(ekg)
    
    # Add the features to the decision tree data
    features2.append(fs)

    # Insert the features to the ekg of the ekg array so they can be
    # used to train the neural network
    f = np.array(list(fs.values()))
    row = np.zeros((len(f), ))
    row[:] = f
    features_temp.append(row)
features_array = np.array(features_temp)

# Replace NaNs from the dataframe array with mean
features2 = pd.DataFrame(data=features2)
features_tree = features.join(features2)
features_tree = features_tree.dropna()
# for col in features_tree.columns:
    # features_tree[col] = features_tree[col].fillna(features_tree[col].median())
features_tree.drop('index', axis=1)

# Downsample all ekgs
down_sampled_ekgs = []
for ekg in ekgs:
    down_sampled_lead = downsample(ekg[1])
    down_sampled_ekgs.append(down_sampled_lead)
down_sampled_ekgs = np.array(down_sampled_ekgs) / 1000

# Merge features with the ekgs, drop any NaNs
down_sampled_ekgs_with_features = []
for i in features_tree.index.values.tolist():
    row = np.zeros((2 + down_sampled_ekgs.shape[1] + features_array.shape[1], ))
    row[2:2+features_array.shape[1]] = features_array[i]
    row[-down_sampled_ekgs.shape[1]:] = down_sampled_ekgs[i]
    down_sampled_ekgs_with_features.append(row)
down_sampled_ekgs_with_features = np.array(down_sampled_ekgs_with_features)
down_sampled_ekgs_with_features[:, :2] = features[['Age', 'Sex']].to_numpy()

print('\nrows of data:', features_tree.shape[0])
print('columns of data:', features_tree.shape[1])

# Train trees
# print('\ntraining baseline decision tree...')
# baseline_tree(features_tree, 'ECG: atrial fibrillation', num_iterations=10)
# print('\ntraining short decision tree...')
# short_tree(features_tree, 'ECG: atrial fibrillation', num_iterations=10)
# print('\ntraining big forest...')
# big_forest(features_tree, 'ECG: atrial fibrillation', num_iterations=10)
# print('\ntraining big forest small tree...')
# big_forest_small_tree(features_tree, 'ECG: atrial fibrillation', num_iterations=10)
# print('\ntraining entropy decision tree...')
# entropy_tree(features_tree, 'ECG: atrial fibrillation', num_iterations=10)
# print('\ntraining log_loss decision tree...')
# log_loss_tree(features_tree, 'ECG: atrial fibrillation', num_iterations=10)

# Train networks
# print('training baseline feature neural network...')
# baseline_feature_network(features_tree[[col for col in features_tree.columns if col != 'ECG: atrial fibrillation']].to_numpy(), features_tree['ECG: atrial fibrillation'].to_numpy(), lr=10e-6)
print('training baseline ekg neural network...')
baseline_feature_network(down_sampled_ekgs_with_features[:, :34], features_tree['ECG: atrial fibrillation'].to_numpy(), lr=10e-6)

# baseline_network(down_sampled_ekgs_with_features, features_tree['ECG: atrial fibrillation'].to_numpy(), lr=0.001)
# print('training cnn neural network')
# cnn(features, ekgs[:, 1:2, :], 'ECG: atrial fibrillation', lr=1e-6)
