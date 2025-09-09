# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:16:51 2025

Script to simulate reinforcement learning model of the bimanual cursor control task

@author: adrianhaith
"""

# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from models import CursorControlEnv, CursorControlLearner
from plotting import plot_value_function, plot_policy
from visualization import CursorLearningVisualizer

np.random.seed(1)

# % Simulate learning
# Create environment
env = CursorControlEnv(radius=.12, motor_noise_std=.05)

# Create learner
participant = CursorControlLearner(
    alpha=.08,
    alpha_nu=0.08,
    sigma=.05,
    seed=1,
    baseline_decay=0.95,
    )

# Initialize the baseline
bsl_states, bsl_rewards, actions = participant.initialize_baseline(env, n_trials=100)
ax = plot_value_function(participant.V, participant)
ax.plot(bsl_states, bsl_rewards, 'o', label='Sampled rewards')

# plot initial policy
mu_init = plot_policy(participant)

# %%
# Run learning for 2000 trials
n_trials = 2600
#n_basis = 36
history = {
    'target_angles': np.zeros(n_trials),
    'actions': np.zeros((n_trials,4)),
    'rewards': np.zeros(n_trials),
    'Ws': np.zeros((n_trials, 4, participant.n_basis)),
    'nus': np.zeros((n_trials, 4)),
    'Vs': np.zeros((n_trials, participant.n_basis)),
    'abs_dir_errors': np.zeros(n_trials)
}

for trial in range(n_trials):
    # store variables at start of trial (for later plotting)
    history['Vs'][trial] = participant.V.copy()
    history['Ws'][trial] = participant.W.copy()
    history['nus'][trial]     = participant.nu.copy()   # <-- copy current ν

    # Get new target angle
    s = env.reset()
    history['target_angles'][trial]=s
    
    # Sample action and get reward
    a, mu, sigma, phi = participant.sample_action(s)
    _, r, _, info = env.step(a)
    
    # Update learner
    participant.update(a, mu, sigma, phi, r)
    
    # Store data for this trial
    history['actions'][trial] = a
    history['rewards'][trial] = r
    history['abs_dir_errors'][trial] = info['abs_directional_error']



# %% ------plot learning time course------
#---------------------

def bin_data(array, bin_size=60):
    """
    Returns average values of input array binned into bins of size block_size
    """
    n_bins = len(array) // bin_size
    trimmed = array[:n_bins * bin_size]  # drop incomplete final block
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    return binned, n_bins, bin_size

time = np.arange(n_trials)
action_labels = ['Lx', 'Ly', 'Rx', 'Ry']

# Compute standard deviations from nu
stds = np.exp(history['nus'])  # shape (n_trials, 4)

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(4, 8), sharex=True)

bin_size = 60
rwd_binned, n_bins, _ = bin_data(history['rewards'], bin_size=bin_size)
bin_centers = bin_size*(np.arange(n_bins)+.5)

# Top panel: Rewards
axs[0].plot(bin_centers, rwd_binned, marker='o', label='Reward')
axs[0].set_ylabel("Reward")
axs[0].set_title("Learning performance")

# Middle panel: Actions
for i in range(4):
    action_binned, _, _ = bin_data(history['actions'][:, i], bin_size=bin_size)
    axs[1].plot(bin_centers, action_binned, marker='o', label=action_labels[i])
axs[1].set_ylabel("Action values")
axs[1].legend()

# Standard deviations (sqrt eigenvalues)

for i in range(4):
    std_binned, _, _ = bin_data(np.exp(history['nus'][:, i]), bin_size=bin_size)
    axs[2].plot(bin_centers, std_binned, marker='o', label=action_labels[i])
axs[2].set_ylabel("Std Dev")
axs[2].set_xlabel("Trial")
axs[2].legend()

# absolute direction error - for comparison with human data
dir_errors_binned, _, _ = bin_data(np.rad2deg(history['abs_dir_errors']), bin_size=bin_size)
axs[3].plot(bin_centers, dir_errors_binned, marker='o', label='|directional_error|',markersize=3)
axs[3].set_yticks([0, 30, 60, 90])
axs[3].set_xlabel("Trial")
axs[3].set_ylabel("Absolute Directional Error")

plt.savefig("learning_timecourse.svg", format="svg", bbox_inches='tight')

plt.tight_layout()
plt.show()

# %% log-log plot
if(1):
    plt.figure(figsize=(5, 4))
    plt.loglog(bin_centers, dir_errors_binned, marker='o', label='Directional Error')
    plt.xlabel('Trial Number')
    plt.ylabel('Directional Error (radians)')
    plt.title('Learning Curve (log-log)')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% plot early/mid/late learning
#plot_learning_progress(history['actions'], history['target_angles'])

#anim = make_animation(history['actions'],history['target_angles'], save_path="cursor_learning.mp4")

# %% -Figure out early learning
#tt=416
#plot_task_snapshot(history['target_angles'][tt],history['actions'][tt])
#plot_policy_update(history['Ws'][tt],history['Ws'][tt+1],history['target_angles'][tt],participant,action=history['actions'][tt])
#ax = plot_value_function(history['Vs'][tt],participant)
#ax.plot(history['target_angles'][tt],history['rewards'][tt], 'o', label='Sampled rewards')
# %%


viz = CursorLearningVisualizer(participant, env, history)

tt=1997
viz.plot_snapshot(2000)
#viz.plot_value_function(tt)
viz.plot_policy_update(participant, tt)
viz.plot_learning_progress([0,700, 1800], window=300)

plt.savefig("endpoint_convergence.svg", format="svg", bbox_inches='tight')

# %%
