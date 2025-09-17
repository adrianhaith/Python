import numpy as np
from numpy.linalg import lstsq
from scipy.special import i0

def compute_von_mises_basis(angles, n_basis=36, kappa=.1):
    centers = np.linspace(0, 2*np.pi, n_basis, endpoint=False)
    phi = np.exp(kappa * (np.cos(angles[:, None] - centers[None, :])))
    phi /= (2 * np.pi * i0(kappa))
    return phi

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def wrap_to_2pi(x):
    return x % (2*np.pi)