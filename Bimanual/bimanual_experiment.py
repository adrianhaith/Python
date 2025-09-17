
# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from models import CursorControlEnv, CursorControlLearner
#from plotting import plot_value_function, plot_policy
from visualization import CursorLearningVisualizer
from utils import compute_von_mises_basis, wrap_to_pi


def run_simulation(n_trials=2500, seed=0):

    np.random.seed(2*seed)

    env = CursorControlEnv(radius=.12, motor_noise_std=.075, discrete_targs=False, seed=seed)
    learner = CursorControlLearner(
        alpha=0.1,
        alpha_nu=0.1,
        sigma=.05,
        seed=seed,
        baseline_decay=0.95,
        kappa=5,
        epsilon=0.4
    )

    learner.initialize_baseline(env, n_trials=1000)

    history = {
        'rewards': np.zeros(n_trials),
        'abs_dir_errors': np.zeros(n_trials),
        'nus': np.zeros((n_trials, 4)),
        'actions': np.zeros((n_trials,4)),
        'targ_dir': np.zeros(n_trials),
        'directional_error': np.zeros(n_trials)
    }

    for t in range(n_trials):
        s = env.reset()
        a, mu, sigma, phi = learner.sample_action(s)
        _, r, _, info = env.step(a)
        learner.update(a, mu, sigma, phi, r)

        history['rewards'][t] = r
        history['abs_dir_errors'][t] = info['abs_directional_error']
        history['nus'][t] = learner.nu.copy()
        history['actions'][t] = a
        history['targ_dir'][t] = s
        history['directional_error'][t] = info['directional_error']

    return history

def bin_data(array, bin_size=60):
    """
    Returns average values of input array binned into bins of size block_size
    """
    n_bins = len(array) // bin_size
    trimmed = array[:n_bins * bin_size]  # drop incomplete final block
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    return binned, n_bins, bin_size



# %% run multiple simulations
n_runs = 20
n_trials = 2200
bin_size = 60

all_binned_rewards = []
all_binned_dir_errors = []

all_targ_dirs = np.zeros((n_runs,n_trials))
all_errors = np.zeros((n_runs,n_trials))

for run in range(n_runs):
    np.random.seed(run)
    hist = run_simulation(n_trials=n_trials, seed=run)
    rwd_binned, n_bins, _ = bin_data(hist['rewards'], bin_size=bin_size)
    dir_binned, _, _ = bin_data(np.rad2deg(hist['abs_dir_errors']), bin_size=bin_size)
    all_binned_rewards.append(rwd_binned)
    all_binned_dir_errors.append(dir_binned)

    all_errors[run]=hist['directional_error']
    all_targ_dirs[run] = hist['targ_dir']
    
    print("run ",run )

all_binned_rewards = np.stack(all_binned_rewards)         # shape (n_runs, n_bins)
all_binned_dir_errors = np.stack(all_binned_dir_errors)

# Compute mean and SEM
mean_rwd = np.mean(all_binned_rewards, axis=0)
sem_rwd = np.std(all_binned_rewards, axis=0) / np.sqrt(n_runs)
std_rwd = np.std(all_binned_rewards, axis=0)

mean_dir = np.mean(all_binned_dir_errors, axis=0)
sem_dir = np.std(all_binned_dir_errors, axis=0) / np.sqrt(n_runs)
std_dir = np.std(all_binned_dir_errors, axis=0)

bin_centers = bin_size * (np.arange(all_binned_rewards.shape[1]) + 0.5)

# %% plot outcomes

plt.figure(figsize=(4, 4))


plt.subplot(2,1,1)
#plt.plot(bin_centers, mean_rwd, label='Reward')
plt.loglog(bin_centers,mean_dir)
plt.fill_between(bin_centers, mean_rwd - std_rwd, mean_rwd + std_rwd, alpha=0.3)
plt.ylabel("Reward")
plt.title("Learning performance (average of multiple runs)")

plt.subplot(2,1,2)
plt.plot(bin_centers, mean_dir, label='|directional error|')
plt.fill_between(bin_centers, mean_dir - std_dir, mean_dir + std_dir, alpha=0.3)
plt.ylabel("Directional Error (deg)")
plt.xlabel("Trial")
plt.yticks([0, 30, 60, 90])
plt.tight_layout()
plt.savefig("averaged_learning_curve.svg", format="svg", bbox_inches='tight')
plt.grid(True)
plt.show()




# %% Now examine bias versus variance decomposition
from numpy.linalg import lstsq
from scipy.special import i0






def analyze_bias_variability_sliding(theta_targets, theta_errors, window=60, n_basis=36, kappa=0.000001):
    #n_trials = len(theta_targets)
    n_windows = n_trials // window

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
        w, _, _, _ = lstsq(Phi, error_win, rcond=None)
        bias_est = Phi @ w
        residuals = error_win - bias_est

        bias_vars[i] = np.var(np.rad2deg(bias_est))
        resid_vars[i] = np.var(np.rad2deg(residuals))
        total_vars[i] = np.var(np.rad2deg(error_win))

        if(0):
            # visualize the errors for this block of trials
            theta_grid = np.linspace(0, 2*np.pi, 500)
            Phi_grid = compute_von_mises_basis(theta_grid, n_basis=n_basis, kappa=kappa)
            predicted_error = Phi_grid @ w  # shape (500,)

            plt.figure(figsize=(6, 4))
            plt.scatter(np.rad2deg(target_win), np.rad2deg(error_win), alpha=0.3, label='Actual errors', s=10)
            plt.plot(np.rad2deg(theta_grid), np.rad2deg(predicted_error), color='black', linewidth=2, label='Fitted bias')
            plt.xlabel('Target direction (deg)')
            plt.ylabel('Directional error (deg)')
            plt.title('Predicted vs Actual Directional Errors')
            plt.axhline(0, color='gray', linestyle='--')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    return bias_vars, resid_vars, total_vars

bias_vars_all = []
resid_vars_all = []
total_vars_all = []
for run in range(n_runs):
    # estimate bias across angles and calculate residual error due to variance
    bias_vars, resid_vars, total_vars = analyze_bias_variability_sliding(theta_targets=all_targ_dirs[run], theta_errors=all_errors[run], window=60)
    bias_vars_all.append(bias_vars)
    resid_vars_all.append(resid_vars)
    total_vars_all.append(total_vars)
# %
scaling = 1#180*180/(np.pi**2)
plt.plot(scaling*np.mean(bias_vars_all, axis=0), color="blue")
plt.plot(scaling*np.mean(resid_vars_all, axis=0), color="red")
plt.plot(scaling*np.mean(total_vars_all, axis=0), color="black")
plt.grid(True)

# %% load human data for comparison
from scipy.io import loadmat

mat = loadmat('human_de_novo_errors.mat')
subjects = mat['subject_data'][0]


bias_vars_human_all = []
resid_vars_human_all = []
total_vars_human_all = []

for subj in range(13):
    bias_vars, resid_vars, total_vars = analyze_bias_variability_sliding(theta_targets=subjects[subj]['target_dir'].squeeze(), theta_errors=subjects[subj]['directional_error'].squeeze(), window=60)

    bias_vars_human_all.append(bias_vars)
    resid_vars_human_all.append(resid_vars)
    total_vars_human_all.append(total_vars)

plt.plot(np.mean(bias_vars_human_all, axis=0), color="blue")
plt.plot(np.mean(resid_vars_human_all, axis=0), color="red")
plt.plot(np.mean(total_vars_human_all, axis=0), color="black")
plt.grid(True)

    
