# %% Script to create Figure 1

import numpy as np
import matplotlib.pyplot as plt

from agent import PGLearnerSimple, PGLearner
from toy_env import Toy2DEnv
from plotting import plot_toy2d_reward_landscape

# Set random number seed
np.random.seed(2)

# --- Initialize environment and learner ---
env = Toy2DEnv()
learner = PGLearnerSimple(init_mean=[0.4, 0.4], init_std=[0.08, 0.04], alpha_mu=0.001, alpha_nu=0.01, alpha_phi=0.01)
learner.initialize_rwd_baseline(env)

# visualize the reward landscape and initial policy
sampled_actions = [learner.select_action() for _ in range(40)]
policycolor = 'yellow'
plot_toy2d_reward_landscape(env, learner=learner, actions=sampled_actions, policycolor=policycolor,outline=True)
#plot_toy2d_reward_landscape(env, learner=learner, policycolor=policycolor,outline=True)
plt.show()

plot_toy2d_reward_landscape(env, learner=learner, actions=sampled_actions, RPE=True, policycolor=policycolor,outline=True)
plt.show()