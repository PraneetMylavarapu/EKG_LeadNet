import pandas as pd
import numpy as np
from ekg import load_ekgs, downsample, merge_ekgs_features
from features import get_features, balance_data
from trees import *
from globals import *
# from networks import *

# Load data
ekgs, features, diagnoses = load_ekgs()
features = features.join(diagnoses['ECG: atrial fibrillation'])

print('num ekgs:', ekgs.shape[0])

# Get equal amount of label=1 and label=0
print('Balancing data...')
features, ekgs = balance_data(features, ekgs)

# Extract features from waveforms
print('Extracting features...')
features_array, features_tree = get_features(ekgs, features)

# Downsample all ekgs
print('Down sampling ekgs...')
down_sampled_ekgs = downsample(ekgs)

# Merge features with the ekgs, drop any NaNs
down_sampled_ekgs_with_features = merge_ekgs_features(down_sampled_ekgs, features_array, features_tree)

print('\nrows of data:', features_tree.shape[0])
print('columns of data:', features_tree.shape[1])

# Train trees
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

# Train networks
# print('training baseline feature neural network...')
# baseline_network(down_sampled_ekgs_with_features[:, :34], features_tree['ECG: atrial fibrillation'].to_numpy(), lr=500e-6)

# print('training baseline feature neural network...')
# baseline_network(down_sampled_ekgs_with_features, features_tree['ECG: atrial fibrillation'].to_numpy(), lr=1e-3)

# print('training cnn neural network')
# cnn(features, ekgs[:, 1, :], 'ECG: atrial fibrillation', lr=5e-6)
