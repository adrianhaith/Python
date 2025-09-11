# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:43:57 2025

@author: adrianhaith
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from arc_task_env import ArcTaskEnv, make_arc_subgoals
from traj_learner import TrajLearner
from wrist_model import WristLDS
from plotting import plot_arc_trials, ArcVisualizer

np.random.seed(2)

# % initialize and check baseline behavior
# create arc task environment
arc_env = ArcTaskEnv(dt=.001) 

# create learner
init_arc_goals = make_arc_subgoals(arc_env.Ng)

# run initial trial to get traject length
arc_env.reset()
_, _, _, inf = arc_env.step(init_arc_goals)
x_traj = inf['trajectory']
NT = np.shape(x_traj)[0]

participant = TrajLearner(Ng=arc_env.Ng,
                          init_goals=init_arc_goals,
                          init_std=0.08,
                          alpha=0.1,
                          alpha_nu=0.1,
                          baseline_decay=.99)

participant.initialize_baseline(arc_env)

# plot baseline behavior
plot_arc_trials(arc_env, participant, n_trials=5)
plt.savefig("initial_trajectories.svg", format="svg", bbox_inches='tight')

# %% now learn improved policy
n_trials = 1000

history = {
    'actions': np.zeros((n_trials,2,6)),
    'means' : np.zeros((n_trials,12)),
    'rewards': np.zeros(n_trials),
    'stds': np.zeros((n_trials, 12)),
    'trajectories': np.zeros((n_trials, NT, 2)),
    'radial_pos' : np.zeros((n_trials, NT))
}


for trial in range(n_trials):
    action = participant.sample_action()
    
    # record mean and std of action distribution that action was sampled from
    history['means'][trial] = participant.init_std * participant.mean_norm
    history['stds'][trial] = participant.init_std * np.sqrt(np.exp(participant.nu))

    _, reward, _, info = arc_env.step(action)
    participant.update(action, reward)

    # update history
    history['rewards'][trial] = reward
    history['actions'][trial] = action
    trajectory = info['trajectory'][:,[0,5]] # position x-y trajectory
    history['trajectories'][trial] = trajectory # x position
    #history['trajectories'][trial,:,1] = info['trajectory'][:,5] # y position
    
    # compute radial position
    radial_pos = np.sqrt((arc_env.radius-trajectory[:,0])**2 + trajectory[:,1]**2)
    history['radial_pos'][trial] = radial_pos

    if trial % 50 == 0:
        print(f"Trial {trial}, Reward: {reward}")

# plot trajectories post-learning
plot_arc_trials(arc_env, participant, n_trials=5)
plt.savefig("late_trajectories.svg", format="svg", bbox_inches='tight')

# %%
# create object to visualize the data
vis = ArcVisualizer(arc_env, participant, history)
vis.plot_trials((0, 10), title="Early trials")
vis.plot_trials((990, 1000), title="Late trials")

# --------------------------------
# %% plot learning timecourse

# helper function to bin data for cleaner plots
def bin_data(array, bin_size=50):
    """
    Returns average values of input array binned into bins of size block_size
    """
    n_bins = len(array) // bin_size
    trimmed = array[:n_bins * bin_size]  # drop incomplete final block
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    return binned, n_bins, bin_size

fig, axs = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

bin_size = 50
rwd_binned, n_bins, _ = bin_data(history['rewards'][:], bin_size=bin_size)
bin_centers = bin_size*(np.arange(n_bins)+.5)

axs[0].plot(bin_centers, rwd_binned, marker='o', label='Reward')
axs[0].set_ylabel("Reward")
axs[0].set_xlabel("Trials")

for i in range(12):
    std_binned, _, _ = bin_data(history['stds'][:,i], bin_size=bin_size)
    axs[1].plot(bin_centers, std_binned, marker='o')
axs[1].set_ylabel("Std Dev")
axs[1].set_xlabel("Trial")
axs[1].legend()


# %% 