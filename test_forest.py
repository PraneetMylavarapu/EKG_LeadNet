from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from ekg import get_ekg_features, load_ekgs

ekgs, features = load_ekgs()
ekgs[0][1]