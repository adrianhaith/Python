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
from plotting import plot_arc_trials

np.random.seed(1)

# % initialize and check baseline behavior
# create arc task environment
arc_env = ArcTaskEnv(dt=.001) 

# create learner
init_arc_goals = make_arc_subgoals(arc_env.Ng)
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
all_rewards = []

for trial in range(n_trials):
    action = participant.sample_action()
    _, reward, _, info = arc_env.step(action)
    participant.update(action, reward)
    all_rewards.append([reward])

    if trial % 50 == 0:
        print(f"Trial {trial}, Reward: {reward}")

# plot trajectories post-learning
plot_arc_trials(arc_env, participant, n_trials=5)
plt.savefig("late_trajectories.svg", format="svg", bbox_inches='tight')
