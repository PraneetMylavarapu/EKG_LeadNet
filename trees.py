from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from random import randint
import pandas as pd
import numpy as np

RANDOM_STATE = 0

def tree_template(data: pd.DataFrame, target: str, n_estimators: int=100, random_state: int=RANDOM_STATE, max_depth: int=None, criterion='gini'):
    feature_labels = data.columns.copy().tolist()
    feature_labels.remove(target)

    X = data[feature_labels]
    y = data[target]

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Create a random forest classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        max_depth=max_depth,
        criterion=criterion
    )
    clf.fit(X_train,y_train)

    # Model accuracy
    y_pred=clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    cnf_matrix = np.zeros((2, 2))
    for y, y_hat in zip(y_test.tolist(), y_pred):
        cnf_matrix[y, y_hat] += 1

    return acc, cnf_matrix

def baseline_tree(data: pd.DataFrame, target: str, print_score: bool=True, num_iterations: int=1):
    best_accuracy = 0
    best_cnf = None
    random_states = [0]
    if num_iterations > 1:
        random_states = [randint(0, 511) for _ in range(num_iterations)]
    
    for state in random_states:
        acc, cnf = tree_template(data, target, random_state=state)
        if acc > best_accuracy:
            best_accuracy = acc
            best_cnf = cnf

    if print_score:
        print(best_cnf)
        print("Accuracy:", best_accuracy)
    
    return best_cnf, best_accuracy

def short_tree(data: pd.DataFrame, target: str, print_score: bool=True, num_iterations: int=1):
    best_accuracy = 0
    best_cnf = None
    random_states = [0]
    if num_iterations > 1:
        random_states = [randint(0, 511) for _ in range(num_iterations)]
    
    for state in random_states:
        acc, cnf = tree_template(data, target, n_estimators=250, max_depth=3, random_state=state)
        if acc > best_accuracy:
            best_accuracy = acc
            best_cnf = cnf

    if print_score:
        print(best_cnf)
        print("Accuracy:", best_accuracy)
    
    return best_cnf, best_accuracy

def big_forest(data: pd.DataFrame, target: str, print_score: bool=True, num_iterations: int=1):
    best_accuracy = 0
    best_cnf = None
    random_states = [0]
    if num_iterations > 1:
        random_states = [randint(0, 511) for _ in range(num_iterations)]
    
    for state in random_states:
        acc, cnf = tree_template(data, target, n_estimators=500, random_state=state)
        if acc > best_accuracy:
            best_accuracy = acc
            best_cnf = cnf

    if print_score:
        print(best_cnf)
        print("Accuracy:", best_accuracy)
    
    return best_cnf, best_accuracy

def big_forest_small_tree(data: pd.DataFrame, target: str, print_score: bool=True, num_iterations: int=1):
    best_accuracy = 0
    best_cnf = None
    random_states = [0]
    if num_iterations > 1:
        random_states = [randint(0, 511) for _ in range(num_iterations)]
    
    for state in random_states:
        acc, cnf = tree_template(data, target, n_estimators=750, max_depth=3, random_state=state)
        if acc > best_accuracy:
            best_accuracy = acc
            best_cnf = cnf

    if print_score:
        print(best_cnf)
        print("Accuracy:", best_accuracy)
    
    return best_cnf, best_accuracy

def entropy_tree(data: pd.DataFrame, target: str, print_score: bool=True, num_iterations: int=1):
    best_accuracy = 0
    best_cnf = None
    random_states = [0]
    if num_iterations > 1:
        random_states = [randint(0, 511) for _ in range(num_iterations)]
    
    for state in random_states:
        acc, cnf = tree_template(data, target, criterion='entropy', random_state=state)
        if acc > best_accuracy:
            best_accuracy = acc
            best_cnf = cnf

    if print_score:
        print(best_cnf)
        print("Accuracy:", best_accuracy)
    
    return best_cnf, best_accuracy

def log_loss_tree(data: pd.DataFrame, target: str, print_score: bool=True, num_iterations: int=1):
    best_accuracy = 0
    best_cnf = None
    random_states = [0]
    if num_iterations > 1:
        random_states = [randint(0, 511) for _ in range(num_iterations)]
    
    for state in random_states:
        acc, cnf = tree_template(data, target, criterion='log_loss', random_state=state)
        if acc > best_accuracy:
            best_accuracy = acc
            best_cnf = cnf

    if print_score:
        print(best_cnf)
        print("Accuracy:", best_accuracy)
    
    return best_cnf, best_accuracy