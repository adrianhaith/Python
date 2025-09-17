import numpy as np
from numpy.linalg import lstsq
from scipy.special import i0
import matplotlib.pyplot as plt

def compute_von_mises_basis(angles, n_basis=36, kappa=.1):
    centers = np.linspace(0, 2*np.pi, n_basis, endpoint=False)
    phi = np.exp(kappa * (np.cos(angles[:, None] - centers[None, :])))
    phi /= (2 * np.pi * i0(kappa))
    return phi

def bin_data(array, bin_size=60):
    """
    Returns average values of input array binned into bins of size block_size
    """
    n_bins = len(array) // bin_size
    trimmed = array[:n_bins * bin_size]  # drop incomplete final block
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    return binned, n_bins, bin_size

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def wrap_to_2pi(x):
    return x % (2*np.pi)

from sklearn.linear_model import Ridge

def ridge_fit(Phi, Y, alpha=0.1):
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(Phi, Y)
    return model.coef_.T  # shape: (n_dims, n_basis)

def analyze_bias_variability_sliding(theta_targets, theta_errors, window=60, n_basis=16, kappa=5):
    #n_trials = len(theta_targets)
    n_windows = np.size(theta_targets) // window

    bias_vars = np.zeros(n_windows)
    resid_vars = np.zeros(n_windows)
    total_vars = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * window
        end = start + window

        target_win = theta_targets[start:end]
        error_win = theta_errors[start:end]
        error_win = wrap_to_pi(error_win)

        Phi = compute_von_mises_basis(target_win, n_basis=n_basis, kappa=kappa)
        w = ridge_fit(Phi, error_win, alpha=1)
        #w, _, _, _ = lstsq(Phi, error_win, rcond=None)
        bias_est = Phi @ w
        residuals = error_win - bias_est

        #plt.plot(w)

        bias_vars[i] = np.var(np.rad2deg(bias_est))
        resid_vars[i] = np.var(np.rad2deg(residuals))
        total_vars[i] = np.var(np.rad2deg(error_win))

        if(0): # use only for visualizing fits
            # visualize the errors for this block of trials
            theta_grid = np.linspace(0, 2*np.pi, 500)
            Phi_grid = compute_von_mises_basis(theta_grid, n_basis=n_basis, kappa=kappa)
            predicted_error = Phi_grid @ w  # shape (500,)

            plt.figure(figsize=(6, 4))
            plt.scatter(np.rad2deg(target_win), np.rad2deg(error_win), alpha=0.3, label='Actual errors', s=10)
            plt.plot(np.rad2deg(theta_grid), np.rad2deg(predicted_error), color='black', linewidth=2, label='Fitted bias')


            # plot a single basis function
            # if(0)
            # for i in range(16):
            #     w_single = np.zeros(16)
            #     w_single[i]=1
            #     prediction_single = Phi_grid @ w_single
            #     plt.plot(np.rad2deg(theta_grid), 10*prediction_single, color="red")

            plt.xlabel('Target direction (deg)')
            plt.ylabel('Directional error (deg)')
            plt.title('Predicted vs Actual Directional Errors')
            plt.axhline(0, color='gray', linestyle='--')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    return bias_vars, resid_vars, total_vars