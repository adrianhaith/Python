# %%
import numpy as np
from scipy.io import loadmat
from numpy.linalg import lstsq
from models import CursorControlLearner
from models import CursorControlEnv
from utils import compute_von_mises_basis, wrap_to_pi, wrap_to_2pi
import matplotlib.pyplot as plt

from plotting import plot_policy, plot_value_function

def fit_human_policy(target_angles, hand_movements, n_basis=16, kappa=5):
    """
    Fit initial policy weights (W_init) and log-std (nu_init) from human data.
    target_angles: shape (n_trials,) in radians
    hand_movements: shape (n_trials, 2), [LH_y, RH_x] endpoint movements
    """
    # Build basis function matrix
    Phi = compute_von_mises_basis(target_angles, n_basis=n_basis, kappa=kappa)  # (n_trials, n_basis)

    # Fit mean movement policy for LH_y and RH_x
    W_human = lstsq(Phi, hand_movements, rcond=None)[0].T  # shape: (2, n_basis)
    W_init = W_human
    # Compute predicted movement magnitude and rescale to match target radius
    #pred_mag = np.linalg.norm(Phi @ W_human.T, axis=1)
    #scale = target_radius / np.mean(pred_mag)
    #W_scaled = scale * W_human

    # Fit residual variance and scale it accordingly
    residuals = hand_movements - Phi @ W_human.T
    var_est = np.var(residuals, axis=0)

    # Initialize W and nu for full 4D action (Lx, Ly, Rx, Ry)
    std_init = np.mean(var_est)
    
    # compare actual and fitted policies
    n_points=200
    angles = np.linspace(0, 2 * np.pi, n_points)
    phi_matrix = np.array([compute_von_mises_basis(s) for s in angles])  # (n_points, n_basis)
    

    return W_init, std_init



# Load human data from .mat file
from pathlib import Path
DATA_PATH = Path('human_data.mat').parent / 'human_data.mat'
mat = loadmat(DATA_PATH)
subj_data = mat['human_data'][0][0]  # adjust indexing as needed

target_angles = wrap_to_2pi(subj_data['target_angles'].squeeze())  # (n_trials,)
hand_movements = subj_data['hand_movements'].T  # (n_trials, 2)


# Fit initial policy
W_init, std_init = fit_human_policy(target_angles, hand_movements, n_basis=16)

# Initialize environment and agent
env = CursorControlEnv(radius=0.12)
participant = CursorControlLearner(alpha=0.1, alpha_nu=0.1, sigma=std_init)

plt.plot(target_angles, hand_movements[:,1], 'o', color="blue")
plt.plot(target_angles, hand_movements[:,2], 'o', color="orange")
plot_policy(participant)

# initial policy
participant.W = W_init.copy() / std_init
#participant.nu = nu_init.copy()
participant.initialize_baseline(env, n_trials=1000)

# visualize intial policy
plot_value_function(participant.V, participant)
mu_init = plot_policy(participant)

# %% Run learning
n_trials = 2000
history = {
    'rewards': np.zeros(n_trials),
    'target_angles': np.zeros(n_trials),
    'actions': np.zeros((n_trials, 4)),
    'nu': np.zeros((n_trials, 4))
}

for t in range(n_trials):
    s = env.reset()
    a, mu, sigma, phi = participant.sample_action(s)
    _, r, _, info = env.step(a)
    participant.update(a, mu, sigma, phi, r)

    history['rewards'][t] = r
    history['target_angles'][t] = s
    history['actions'][t] = a
    history['nu'][t] = participant.nu.copy()

# Save or visualize results here as needed