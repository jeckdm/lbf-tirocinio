import numpy as np

def shuffle(X, y, seed = 0):
    """ Ritorna una permutazione di X,y generata con il seed in input"""

    np.random.seed(seed)
    perm = np.random.permutation(len(X))

    X = X[perm]
    y = y[perm]

    return X, y