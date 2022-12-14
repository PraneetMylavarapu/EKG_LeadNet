from ekg import load_ekgs, one_interval, downsample
import matplotlib.pyplot as plt
import numpy as np



ekgs, features, diagnoses = load_ekgs()
features = features.join(diagnoses['ECG: atrial fibrillation'])
test = one_interval(ekgs[0][1], downsample=False)
test = downsample(test)
plt.plot(test)
plt.savefig('test.png')
