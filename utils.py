import numpy as np
from math import floor

# Learning rates

def Time_Step(init, k, rate, t, T):
    '''
    Decays at each iterations
    '''
    return init / (1 + (t + k * T) * rate)


def Step(init, k, rate, *kargs):
    '''
    Decays at step
    '''
    return init / (1 + k * rate)


def Exponential(init, k, rate, t, T):
    '''
    Decays at each iterations exponentialy
    '''
    return init * np.exp(-(t + k * T) * rate)

# Compute S

def Compute_S(d, k, T, updates):
    S = np.zeros(shape=(d, k))
    for i in range(k):
        S[:, i] = k / T * np.sum([updates[_] for _ in range(int(floor(T / k) * i), int(floor(T / k) * (i + 1)))])
    return S