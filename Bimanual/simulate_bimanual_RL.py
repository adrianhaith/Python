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

from models import CursorControlEnv, CursorControlLearner
from plotting import plot_task_snapshot, plot_policy_update, plot_learning_progress, make_animation, plot_value_function, plot_policy

np.random.seed(1)

# % Simulate learning
# Create environment
env = CursorControlEnv(radius=.12)

# Create learner
participant = CursorControlLearner(
    alpha=0.01,
    alpha_nu=0.01,
    sigma=.05,
    seed=1,
    baseline_decay=0.95,
    )

# Initialize the baseline
bsl_states, bsl_rewards = participant.initialize_baseline(env, n_trials=100)
ax = plot_value_function(participant.V, participant)
ax.plot(bsl_states, bsl_rewards, 'o', label='Sampled rewards')

# plot initial policy
plot_policy(participant)

# %%
# Run learning for 2000 trials
n_trials = 1000
#n_basis = 36
actions = np.zeros((n_trials, 4))
target_angles = np.zeros(n_trials)
rewards = np.zeros(n_trials)
nus     = np.zeros((n_trials, 4))   # <-- store log‐eigs
W_pres = np.zeros((n_trials, 4,participant.n_basis))
W_posts = np.zeros((n_trials, 4, participant.n_basis))
Vs = np.zeros((n_trials, participant.n_basis))


for trial in range(n_trials):
    # store variables (for later plotting)
    Vs[trial] = participant.V.copy()
    W_pres[trial] = participant.W.copy()
    

    # Get new target angle
    s = env.reset()
    target_angles[trial]=s
    
    # Sample action and get reward
    a, mu, sigma, phi = participant.sample_action(s)
    _, r, _, _ = env.step(a)
    
    # Update learner
    participant.update(a, mu, sigma, phi, r)
    
    # Store data for this trial
    W_posts[trial] = participant.W.copy()
    actions[trial] = a
    rewards[trial] = r
    nus[trial]     = participant.nu.copy()   # <-- copy current ν

    

# %% plot outcome
n_trials = len(rewards)
time = np.arange(n_trials)
action_labels = ['Lx', 'Ly', 'Rx', 'Ry']

# Compute standard deviations from nu
stds = np.exp(nus)  # shape (n_trials, 4)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Top panel: Rewards
axs[0].plot(time, rewards, label='Reward')
axs[0].set_ylabel("Reward")
axs[0].set_title("Learning performance")
axs[0].grid(True)

# Middle panel: Actions
for i in range(4):
    axs[1].plot(time, actions[:, i], label=action_labels[i])
axs[1].set_ylabel("Action values")
axs[1].legend()
axs[1].grid(True)

# Bottom panel: Standard deviations (sqrt eigenvalues)
for i in range(4):
    axs[2].plot(time, stds[:, i], label=action_labels[i])
axs[2].set_ylabel("Std Dev")
axs[2].set_xlabel("Trial")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

# %% Visualize trial endpoints
trial_range = range(0, n_trials)
cmap = plt.cm.hsv  # Cyclic colormap

# Determine number of targets automatically
n_targets = len(np.unique(target_angles))
canonical_angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)

# Normalize angle to [0, 1) for color mapping
def angle_to_color(angle):
    return cmap((angle % (2 * np.pi)) / (2 * np.pi))

# Begin plot
fig, ax = plt.subplots(figsize=(3, 3))

# Plot central start position
ax.plot(0, 0, 'ko', markersize=4)

# Plot targets
for angle in canonical_angles:
    x, y = env.radius * np.cos(angle), env.radius * np.sin(angle)
    ax.plot(x, y, 'o', color=angle_to_color(angle), markersize=8)

# Plot endpoints
for t in trial_range:
    a = actions[t]
    theta = target_angles[t]
    endpoint = [a[1], a[2]]  # Ly = cursor x, Rx = cursor y
    ax.plot(*endpoint, 'o', color=angle_to_color(theta), markersize=3, alpha=0.6)

# Aesthetic adjustments
ax.set_aspect('equal')
#ax.axis('off')  # Hide all axis lines, ticks, and labels
plt.tight_layout()
plt.show()

# %% plot early/mid/late learning
plot_learning_progress(actions, target_angles)

anim = make_animation(actions,target_angles, save_path="cursor_learning.mp4")

# %% -Figure out early learning
tt=416
plot_task_snapshot(target_angles[tt],actions[tt])
plot_policy_update(W_pres[tt],W_posts[tt],target_angles[tt],participant,action=actions[tt])
ax = plot_value_function(Vs[tt],participant)
ax.plot(target_angles[tt],rewards[tt], 'o', label='Sampled rewards')
# %%
