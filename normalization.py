import numpy as np

def z_score(x):
    n = x[0].shape[0]
    for i in range(n):
        mean = np.mean(x[:,i])
        std = np.std(x[:,i])
        x[:,i] = (x[:,i]-mean)/std
    return x
