
# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from models import CursorControlEnv, CursorControlLearner
from plotting import plot_value_function, plot_policy
from visualization import CursorLearningVisualizer

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
        epsilon=0.5
    )

    learner.initialize_baseline(env, n_trials=1000)

    history = {
        'rewards': np.zeros(n_trials),
        'abs_dir_errors': np.zeros(n_trials),
        'nus': np.zeros((n_trials, 4)),
        'actions': np.zeros((n_trials,4))
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
n_trials = 2600
bin_size = 60

all_rewards = []
all_dir_errors = []

for run in range(n_runs):
    np.random.seed(run)
    hist = run_simulation(n_trials=n_trials, seed=run)
    rwd_binned, n_bins, _ = bin_data(hist['rewards'], bin_size=bin_size)
    dir_binned, _, _ = bin_data(np.rad2deg(hist['abs_dir_errors']), bin_size=bin_size)
    all_rewards.append(rwd_binned)
    all_dir_errors.append(dir_binned)
    print("run",{run})

all_rewards = np.stack(all_rewards)         # shape (n_runs, n_bins)
all_dir_errors = np.stack(all_dir_errors)

# Compute mean and SEM
mean_rwd = np.mean(all_rewards, axis=0)
sem_rwd = np.std(all_rewards, axis=0) / np.sqrt(n_runs)
std_rwd = np.std(all_rewards, axis=0)

mean_dir = np.mean(all_dir_errors, axis=0)
sem_dir = np.std(all_dir_errors, axis=0) / np.sqrt(n_runs)
std_dir = np.std(all_dir_errors, axis=0)

bin_centers = bin_size * (np.arange(all_rewards.shape[1]) + 0.5)

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
