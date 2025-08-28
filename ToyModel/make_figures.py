# %% Script to create Figure 1

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from agent import PGLearnerSimple, PGLearner
from toy_env import Toy2DEnv
from plotting import plot_covariance_ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set random number seed
np.random.seed(2)

# --- Initialize environment and learner ---
env = Toy2DEnv()
learner = PGLearnerSimple(init_mean=[0.4, 0.4], init_std=[0.08, 0.04], alpha_mu=0.001, alpha_nu=0.002, alpha_phi=0.002)
learner.initialize_rwd_baseline(env)

# ---- First Figure: reward heatmap and policy + samples
#  set figure and define colorscheme
fig, ax = plt.subplots(figsize=(6, 6))
# Create a divider to make space for colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

policy_color = 'chartreuse'
rwd_colormap = 'gray'
rpe_colormap = 'coolwarm'

# visualize the reward landscape and initial policy
sampled_actions = [learner.select_action() for _ in range(40)]

# plot the reward landscape and sampled actions
# -- Reward heatmap
U1, U2, R = env.get_reward_grid(resolution=300)
vlim = np.abs(R).max()
heatmap = ax.pcolormesh(U1, U2, R, shading='auto', cmap=rwd_colormap, alpha=1, rasterized=True)
cbar = plt.colorbar(heatmap, cax=cax)

# -- Policy
mean = learner.mean
cov = learner.cov
plot_covariance_ellipse(mean, cov, ax, color=policy_color, outline=True) 

# -- sampled actions
actions = np.array(sampled_actions)
ax.scatter(actions[:, 0], actions[:, 1], marker='o', color=policy_color, edgecolor='black', s=20, alpha=1)

ax.set_xlim(env.u1_range)
ax.set_ylim(env.u2_range)
ax.set_xlabel('u1')
ax.set_ylabel('u2')
ax.set_title('Reward Landscape')
ax.set_aspect('equal')
#plt.savefig("R_heatmap.eps", format="eps", bbox_inches='tight')
plt.savefig("R_heatmap.svg", format="svg", bbox_inches='tight')
plt.show()


# --- Second Figure: RPE heatmap, plus a single sample
fig2, ax2 = plt.subplots(figsize=(6, 6))
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)

# RPE heatmap
RPE = R - learner.rwd_baseline
vlim = np.abs(RPE).max()
heatmap = ax2.pcolormesh(U1, U2, RPE, shading='auto', cmap=rpe_colormap, alpha=1, vmin=-vlim, vmax=vlim, rasterized=True)
cbar = plt.colorbar(heatmap, cax=cax)

# plot policy + mean
plot_covariance_ellipse(mean, cov, ax2, color=policy_color, outline=True)

# plot a single sample
#action = np.array([0.5, 0.39]) # good action with +ve RPE
action = np.array([0.3, 0.41]) # 'bad' sample with -ve RPE
ax2.plot(action[0], action[1], marker='o', markersize=4.5, markerfacecolor=policy_color, markeredgewidth=1, markeredgecolor='black')

# now, update the policy according to the action, and replot the policy
_, reward, _, _ = env.step(action)
learner.update(action, reward)

# re-plot the policy
plot_covariance_ellipse(learner.mean, learner.cov, ax2, color=policy_color, outline=True)


ax2.set_xlim(env.u1_range)
ax2.set_ylim(env.u2_range)
ax2.set_xlabel('u1')
ax2.set_ylabel('u2')
ax2.set_title('RPE Landscape')
ax2.set_aspect('equal')


#plt.savefig("RPE_heatmap1.eps", format="eps", bbox_inches='tight')
plt.savefig("RPE_heatmap_bad_action.svg", format="svg", bbox_inches='tight')
plt.show()


# run the algorithm for 3000 trials then replot
n_trials = 3000
for trial in range(n_trials):
    action = learner.select_action()
    _, reward, _, _ = env.step(action)
    learner.update(action, reward)

#  set figure and define colorscheme
fig, ax3 = plt.subplots(figsize=(6, 6))
# Create a divider to make space for colorbar
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.1)

policy_color = 'chartreuse'
rwd_colormap = 'gray'
rpe_colormap = 'coolwarm'

# visualize the reward landscape and initial policy
sampled_actions = [learner.select_action() for _ in range(40)]

# plot the reward landscape and sampled actions
# -- Reward heatmap
U1, U2, R = env.get_reward_grid(resolution=300)
vlim = np.abs(R).max()
heatmap = ax3.pcolormesh(U1, U2, R, shading='auto', cmap=rwd_colormap, alpha=1, rasterized=True)
cbar = plt.colorbar(heatmap, cax=cax)

# -- Policy
mean = learner.mean
cov = learner.cov
plot_covariance_ellipse(mean, cov, ax3, color=policy_color, outline=True) 

# -- sampled actions
actions = np.array(sampled_actions)
ax3.scatter(actions[:, 0], actions[:, 1], marker='o', color=policy_color, edgecolor='black', s=20, alpha=1)

ax3.set_xlim(env.u1_range)
ax3.set_ylim(env.u2_range)
ax3.set_xlabel('u1')
ax3.set_ylabel('u2')
ax3.set_title('Reward Landscape')
ax3.set_aspect('equal')
#plt.savefig("R_heatmap.eps", format="eps", bbox_inches='tight')
plt.savefig("R_heatmap_post.svg", format="svg", bbox_inches='tight')
plt.show()