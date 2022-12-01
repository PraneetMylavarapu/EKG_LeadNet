from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np


def baseline_tree(data: pd.DataFrame, target: str):
    feature_labels = data.columns.copy().tolist()
    feature_labels.remove(target)

    X = data[feature_labels]
    y = data[target]

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)

    # Model accuracy
    y_pred=clf.predict(X_test)
    cnf_matrix = np.zeros((2, 2))
    for y, y_hat in zip(y_test.tolist(), y_pred):
        cnf_matrix[y, y_hat] += 1
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))