from ekg import load_ekgs
import matplotlib.pyplot as plt
import numpy as np



ekgs, features, diagnoses = load_ekgs()
features = features.join(diagnoses['ECG: atrial fibrillation'])
