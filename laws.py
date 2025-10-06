import numpy as np


def refract(I, N, n1, n2):
    I = I / np.linalg.norm(I)
    N = N / np.linalg.norm(N)
    eta = n1 / n2
    cos_i = -np.dot(I, N)
    k = 1 - eta ** 2 * (1 - cos_i ** 2)

    if k < 0:
        return None

    T = eta * I + (eta * cos_i - np.sqrt(k)) * N

    T = T / np.linalg.norm(T)

    return T


def reflect(I, N):
    I = I / np.linalg.norm(I)
    N = N / np.linalg.norm(N)
    return I - 2 * np.dot(I, N) * N
