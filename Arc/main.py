#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:43:57 2025

@author: adrianhaith
"""

import numpy as np
import matplotlib.pyplot as plt
from arc_task_env import ArcTaskEnv, make_arc_subgoals
from traj_learner import TrajLearner
from wrist_model import WristLDS
from utils import plot_arc_trials

np.random.seed(2)

# %% initialize and check baseline behavior
# create arc task environment
arc_env = ArcTaskEnv(dt=.001) 

# create learner
init_arc_goals = make_arc_subgoals(arc_env.Ng)
participant = TrajLearner(Ng=arc_env.Ng,
                          init_goals=init_arc_goals,
                          init_std=0.006,
                          alpha=0.0005,
                          alpha_nu=0.05,
                          baseline_decay=.99)
participant.initialize_baseline(arc_env)

# plot baseline behavior
plot_arc_trials(arc_env, participant, n_trials=5)


# %% debug by giving a known action, identical to Matlab implementation with fixed seed
#action_matlab = np.array([[.1433, .2160, .4221, .4473, .3962, .4164],[.1692, .2649, .2160, .1537, .0080, .0690]])
#action_matlab = np.array([[.1291, .1682, .3449, .3701, .3627, .4164], [.1473, .2437, .2283, .2233, .1330, .0690]])
#_, reward, _, info = arc_env.step(action_matlab)
#participant.update(action_matlab, reward)

# %% now learn improved policy
n_trials = 2000
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
