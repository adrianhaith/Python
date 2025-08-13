# %%
import numpy as np
import matplotlib.pyplot as plt

from agent import PGLearner
from toy_env import Toy2DEnv
from plotting import plot_toy2d_reward_landscape

# Set random number seed
np.random.seed(1)

# --- Initialize environment and learner ---
env = Toy2DEnv()
learner = PGLearner(init_mean=[0.5, 0.4], init_std=[0.08, 0.04], alpha=1, alpha_phi=0.0, alpha_nu=0.0)
learner.initialize_rwd_baseline(env)

# visualize the reward landscape and initial policy
sampled_actions = [learner.select_action() for _ in range(50)]
plot_toy2d_reward_landscape(env, learner=learner, actions=sampled_actions)
plt.show()


# %%
# --- Run a few trials ---

n_trials = 5
actions = np.zeros((n_trials, 2))
rewards = np.zeros(n_trials)
mus     = np.zeros((n_trials,2))
nus     = np.zeros((n_trials, 2))   # <-- store log‐eigs
phis    = np.zeros(n_trials) 
rwd_baselines = np.zeros(n_trials)



for trial in range(n_trials):

    rwd_baselines[trial] = learner.rwd_baseline


    action = learner.select_action()
    action = [.2, 0.6]
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

