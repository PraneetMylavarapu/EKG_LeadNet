from networks import baseline_network
from random import random
import numpy as np

data = []
labels = []
for _ in range(300):
    data.append([random()*0.75, random()*0.75])
    data.append([1-random()*0.75, 1-random()*0.75])
    labels.append(1)
    labels.append(0)

data = np.array(data)
labels = np.array(labels)

baseline_network(data, labels)