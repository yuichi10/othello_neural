import numpy as np


def load_data():
    X = np.loadtxt("data/X.txt", delimiter=" ")
    Y = np.loadtxt("data/Y.txt", delimiter=" ")
    return X.T, Y.T


X, Y = load_data()

print(X.shape)
print(X)
print(Y.shape)
print(Y)
