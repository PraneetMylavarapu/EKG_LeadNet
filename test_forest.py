from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from ekg import get_ekg_features_test, load_ekgs

ekgs, features = load_ekgs()

# Extract features from waveforms
features2 = []
for ekg in ekgs:
    features2.append(get_ekg_features_test(ekg))
features2 = pd.DataFrame(data=features2)
features3 = features.join(features2)