from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
import pandas as pd
import numpy as np


# Consts
FEATURE_LABELS = ['sepal length', 'sepal width', 'petal length', 'petal width']
PREDICTION_LABEL = 'species'

# Load the ekg data
ekgs = load_ekgs()

# Extract the ekg features
data = get_ekg_features(None, None)
X = data[FEATURE_LABELS]
y = data[PREDICTION_LABEL]

# Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

# Model accuracy
y_pred=clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))