# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:54:38 2025

@author: adrianhaith
"""

# script to simulate skittles

# first, import needed libraries
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from plotting import plot_policy_snapshot
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from models import SkittlesEnv, SkittlesLearner

make_animation = False

# %% Simulate learning

# Set random number seed
np.random.seed(1)

env = SkittlesEnv(target=[.6, .3])

participant = SkittlesLearner(
    init_mean=[30, 1.5],
    init_std=[8, .5],
    alpha=0.05,
    alpha_nu=0.05,
    alpha_phi=0.05,
    rwd_baseline_decay=0.99
)


ax, out = env.plot_sample_trajectories(n_samples=1)

n_trials = 1200
actions = np.zeros((n_trials, 2))
rewards = np.zeros(n_trials)
mus = np.zeros((n_trials, 2))
nus     = np.zeros((n_trials, 2))   # <-- store log‐eigs
phis    = np.zeros(n_trials) 
cov_mats = np.zeros((n_trials, 2, 2))
rwd_baselines = np.zeros(n_trials)

participant.initialize_rwd_baseline(env)

# %%
for t in range(n_trials):

    rwd_baselines[t] = participant.rwd_baseline
    
    _, _ = env.reset()
    a = participant.select_action()
    _, r, _, _, _ = env.step(a)
    participant.update(a, r)
    
    actions[t] = a
    rewards[t] = r
    mus[t]     = participant.mu
    nus[t]     = participant.nu   # <-- copy current ν
    phis[t]    = participant.phi
    cov_mats[t] = participant.covariance


participant.covariance

# %% plot outcome of simulations
bin_size = 10
n_bins   = n_trials // bin_size

def bin_mean(x):
    x = x[:n_bins * bin_size]
    return x.reshape(n_bins, bin_size).mean(axis=1)

angles_deg     = np.rad2deg(actions[:,0])
velocities     = actions[:,1]
rewards_binned = bin_mean(rewards)

lams = np.exp(nus)  # eigenvalues = exp(ν)
lam1_binned = bin_mean(lams[:,0])
lam2_binned = bin_mean(lams[:,1])

phis_deg       = np.rad2deg(phis)

# ——————————————————————
# 3) Plot all five panels
# ——————————————————————
fig, axs = plt.subplots(6,1, figsize=(10,10), sharex=True)

axs[0].plot( bin_mean(angles_deg) )
axs[0].set_ylabel("Angle (°)")
axs[0].set_title("Mean Launch Angle")

axs[1].plot( bin_mean(velocities), color='orange' )
axs[1].set_ylabel("Velocity (m/s)")
axs[1].set_title("Mean Launch Velocity")

axs[2].plot( rewards_binned, color='black' )
axs[2].plot( bin_mean(rwd_baselines), color='red')
axs[2].set_ylabel("Reward")
axs[2].set_title("Mean Reward")

axs[3].plot( lam1_binned, label="λ₁ (angle var)", color='blue' )
axs[3].plot( lam2_binned, label="λ₂ (vel var)", color='red' )
axs[3].set_ylabel("Eigenvalues")
axs[3].set_title("Covariance Eigenvalues")
axs[3].legend()

axs[4].plot( bin_mean(phis_deg), color='purple' )
axs[4].set_ylabel("Covariance Angle")
axs[4].set_xlabel("Trial Bin (10 trials)")
print("made plots")

plt.tight_layout()

# Convert angles to degrees
actions_deg = np.copy(actions)
actions_deg[:, 0] = np.rad2deg(actions[:, 0])

# Split actions
first_100 = actions_deg[:100]
last_100 = actions_deg[-100:]

# %%---Log-log plot to test for power-law learning
from scipy.stats import linregress

# --- Parameters ---
bin_size = 100
n_bins = n_trials // bin_size
trial_bins = np.arange(1, n_trials + 1, bin_size)

# --- Bin and average rewards ---
reward_binned = -np.mean(rewards[:n_bins * bin_size].reshape(n_bins, bin_size), axis=1)

# Avoid issues with log(0) by shifting rewards if needed
eps = 1e-6
reward_binned = np.maximum(reward_binned, eps)

# --- Log-log transform + omit first bin
log_trials = np.log10(trial_bins[1:])
log_rewards = np.log10(reward_binned[1:])

# --- Optional: Linear regression in log-log space ---
slope, intercept, r_value, p_value, std_err = linregress(log_trials, log_rewards)
fit_line = slope * log_trials + intercept

# --- Plot ---
plt.figure(figsize=(6, 4))
plt.plot(log_trials, log_rewards, 'o-', label='Log-Log Data')
plt.plot(log_trials, fit_line, 'r--', label=f'Fit: slope={slope:.2f}')
plt.xlabel("log₁₀(Trial)")
plt.ylabel("log₁₀(Reward)")
plt.title("Log-Log Plot of Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%---Plot heatmap with two sets of actions
A_deg, V, R = env.compute_reward_grid(return_degrees = True)
plt.figure(figsize=(6, 6))
plt.pcolormesh(A_deg, V, R, shading='auto', cmap='gray', alpha=0.9)
plt.scatter(first_100[:, 0], first_100[:, 1], color='blue', s=10, label='First 100 Trials')
plt.scatter(last_100[:, 0], last_100[:, 1], color='red', s=10, label='Last 100 Trials')

plt.xlabel("Launch Angle (degrees)")
plt.ylabel("Velocity (m/s)")
plt.title("Skittles Task: Action Samples Over Learning")
plt.xlim(A_deg.min(), A_deg.max())
plt.ylim(V.min(), V.max())
plt.legend()
plt.grid(True)
plt.tight_layout()

# %%
if(make_animation):
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display
    import numpy as np

    # --- Animation setup ---
    frame_size = 100
    step_size = 20
    n_frames = (n_trials - frame_size) // step_size + 1

    fig, ax = plt.subplots(figsize=(6, 6))
    c = ax.pcolormesh(A_deg, V, R, shading='auto', cmap='gray', alpha=0.9)

    ax.set_xlim(A_deg.min(), A_deg.max())
    ax.set_ylim(V.min(), V.max())
    ax.set_xlabel("Launch Angle (degrees)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Learning Trajectory")

    sc = ax.scatter([], [], color='red', s=10, zorder=3)

    def update(frame):
        start = frame * step_size
        end = start + frame_size
        batch = actions_deg[start:end]
        sc.set_offsets(np.atleast_2d(batch))
        ax.set_title(f"Trials {start+1}–{end}")
        return sc,

    print("making animation...")
    ani = FuncAnimation(fig, update, frames=n_frames, interval=25, blit=False)

    plt.tight_layout()
    plt.close(fig)
    display(HTML(ani.to_jshtml()))
    print("done")




# %% ---- Plot evolution of training in a single panel ---------------------------
# -------------------------------


from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, PowerNorm

# Color map setup
cmap = plt.cm.summer_r  # or 'plasma', 'inferno', etc.
norm = PowerNorm(gamma=.5, vmin=0, vmax=n_trials)  # normalize trial index for colormap


fig, ax = plt.subplots(figsize=(5, 4))

rwd_norm = PowerNorm(gamma=5, vmin=-1, vmax=0)
rwd_map = ax.pcolormesh(A_deg, V, R, cmap='Greys_r', alpha=0.9, norm=rwd_norm, rasterized=True)
cbar = plt.colorbar(rwd_map)

snapshot_step_size = 2000

#snapshot_trials = np.arange(snapshot_step_size-1, n_trials+1, snapshot_step_size)
snapshot_trials = np.array([0, 499, 999, 1999, 3999, 7999])
snapshot_trials = np.concatenate((np.array([0]), snapshot_trials))

colors = [cmap(norm(t)) for t in snapshot_trials] # color to use for each snapshot

for trial, color in zip(snapshot_trials, colors):
    mu = mus[trial]
    nu = nus[trial]
    phi = phis[trial]
    cov = cov_mats[trial]
    plot_policy_snapshot(ax, mu, cov, color)

# Formatting
ax.set_xlim(A_deg.min(), 100)
ax.set_ylim(V.min(), 4)
ax.set_xlabel("Launch Angle (degrees)")
ax.set_ylabel("Velocity (m/s)")

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


plt.tight_layout()
plt.savefig("policy_evolution.svg", format="svg", bbox_inches='tight')
plt.show()

# %% --- make the heatmap
height = 200
width = 10
gradient = np.linspace(R.min(), 0, height).reshape(-1, 1)
gradient = np.repeat(gradient, width, axis=1)  # make it wide enough

# --- Plot it as a heatmap ---
fig, ax = plt.subplots(figsize=(1.2, 4))
im = ax.imshow(gradient, cmap='Greys_r', norm=rwd_norm, aspect='auto', origin='lower')

# Add custom ticks on the y-axis
tick_vals = np.linspace(0, 1, 5)
tick_pos = np.linspace(0, height-1, len(tick_vals))
ax.set_yticks(tick_pos)
ax.set_yticklabels([f"{v:.2f}" for v in tick_vals])
ax.set_xticks([])
ax.set_ylabel("Reward")

# Remove axes for clean visual
#ax.axis('off')

# Save or use inline as a "colorbar-style" gradient
plt.tight_layout()
plt.savefig("rwd_colorbar.svg", format="svg", bbox_inches='tight')
plt.show()





# %% Conduct TNC analysis
from TNC import TNCCost

# actions: (n_trials, 2) array of [release_angle, release_velocity]
block_size = 60
n_blocks = actions.shape[0] // block_size

tnc = TNCCost(env)

all_results = []
for b in range(n_blocks):
    print("block")
    block_actions = actions[b*block_size : (b+1)*block_size]
    results = tnc.compute_all(block_actions)
    all_results.append(results)

# %%
plt.figure(figsize=(8,5))

plt.plot(results["T-Cost"], label="T-Cost")
plt.plot(results["N-Cost"], label="N-Cost")
plt.plot(results["C-Cost"], label="C-Cost")

plt.xlabel("Block (60 trials each)")
plt.ylabel("Cost (mean error difference)")
plt.title("TNC-Cost decomposition across practice")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
