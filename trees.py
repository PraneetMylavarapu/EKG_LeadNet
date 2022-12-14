from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np

RANDOM_STATE = 0

def baseline_tree(data: pd.DataFrame, target: str):
    feature_labels = data.columns.copy().tolist()
    feature_labels.remove(target)

    X = data[feature_labels]
    y = data[target]

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf.fit(X_train,y_train)

    # Model accuracy
    y_pred=clf.predict(X_test)
    cnf_matrix = np.zeros((2, 2))
    for y, y_hat in zip(y_test.tolist(), y_pred):
        cnf_matrix[y, y_hat] += 1
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    return clf

def short_tree(data: pd.DataFrame, target: str):
    feature_labels = data.columns.copy().tolist()
    feature_labels.remove(target)

    X = data[feature_labels]
    y = data[target]

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=250, max_depth=3, random_state=RANDOM_STATE)
    clf.fit(X_train,y_train)

    # Model accuracy
    y_pred=clf.predict(X_test)
    cnf_matrix = np.zeros((2, 2))
    for y, y_hat in zip(y_test.tolist(), y_pred):
        cnf_matrix[y, y_hat] += 1
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    return clf

def big_forest(data: pd.DataFrame, target: str):
    feature_labels = data.columns.copy().tolist()
    feature_labels.remove(target)

    X = data[feature_labels]
    y = data[target]

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE)
    clf.fit(X_train,y_train)

    # Model accuracy
    y_pred=clf.predict(X_test)
    cnf_matrix = np.zeros((2, 2))
    for y, y_hat in zip(y_test.tolist(), y_pred):
        cnf_matrix[y, y_hat] += 1
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    return clf

def entropy_tree(data: pd.DataFrame, target: str):
    feature_labels = data.columns.copy().tolist()
    feature_labels.remove(target)

    X = data[feature_labels]
    y = data[target]

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=RANDOM_STATE)
    clf.fit(X_train,y_train)

    # Model accuracy
    y_pred=clf.predict(X_test)
    cnf_matrix = np.zeros((2, 2))
    for y, y_hat in zip(y_test.tolist(), y_pred):
        cnf_matrix[y, y_hat] += 1
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    return clf

def log_loss_tree(data: pd.DataFrame, target: str):
    feature_labels = data.columns.copy().tolist()
    feature_labels.remove(target)

    X = data[feature_labels]
    y = data[target]

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, criterion='log_loss', random_state=RANDOM_STATE)
    clf.fit(X_train,y_train)

    # Model accuracy
    y_pred=clf.predict(X_test)
    cnf_matrix = np.zeros((2, 2))
    for y, y_hat in zip(y_test.tolist(), y_pred):
        cnf_matrix[y, y_hat] += 1
    print(cnf_matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    return clf