# %%
import numpy as np
import matplotlib.pyplot as plt

from agent import PGLearnerSimple, PGLearner
from toy_env import Toy2DEnv
from plotting import plot_toy2d_reward_landscape

# Set random number seed
np.random.seed(2)

# --- Initialize environment and learner ---
env = Toy2DEnv()
learner = PGLearnerSimple(init_mean=[0.4, 0.5], init_std=[0.1, 0.05], alpha_mu=0.001, alpha_nu=0.01, alpha_phi=0.01)
learner.initialize_rwd_baseline(env)

# visualize the reward landscape and initial policy
sampled_actions = [learner.select_action() for _ in range(50)]
plot_toy2d_reward_landscape(env, learner=learner, actions=sampled_actions)
plt.show()


# %%
# --- Run a few trials ---

n_trials = 5000
actions = np.zeros((n_trials, 2))
rewards = np.zeros(n_trials)
mus     = np.zeros((n_trials,2))
nus     = np.zeros((n_trials, 2))   # <-- store log‐eigs
phis    = np.zeros(n_trials) 
rwd_baselines = np.zeros(n_trials)



for trial in range(n_trials):

    rwd_baselines[trial] = learner.rwd_baseline


    action = learner.select_action()
    _, reward, _, _ = env.step(action)
    learner.update(action, reward)

    actions[trial] = action
    rewards[trial] = reward
    nus[trial]     = learner.nu   # <-- copy current ν
    mus[trial]     = learner.mean
    phis[trial]    = learner.phi

# --- Plot reward over time ---
plt.plot(rewards)
plt.xlabel("Trial")
plt.ylabel("Reward")
plt.title("Reward over time in Toy2DEnv")
plt.show()

plt.plot(mus)
plt.xlabel("Trial")
plt.ylabel("Mu")
plt.title("Mean action over time in Toy2DEnv")
plt.show()

plt.plot(nus)
plt.xlabel("Trial")
plt.ylabel("Nu")
plt.title("action STD over time in Toy2DEnv")
plt.show()

sampled_actions = [learner.select_action() for _ in range(50)]
plot_toy2d_reward_landscape(env, learner=learner, actions=sampled_actions)
plt.show()

# %%
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection

# Assume env is already created:
# env = Toy2DEnv()

U1, U2, R = env.get_reward_grid(resolution=100)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
ax.plot_surface(U1, U2, R, cmap='viridis', edgecolor='none', alpha=0.9)

# Labels
ax.set_xlabel('u1')
ax.set_ylabel('u2')
ax.set_zlabel('Reward')
ax.set_title('Reward Landscape (3D)')

plt.show()