# %% 
import numpy as np
import pandas as pd

from models import SkittlesEnv, SkittlesLearner
from TNC import TNCCost

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

def run_experiment(env, seed, init_mean, n_trials=3000, block_size=60):
    """
    Run one simulation of Skittles learner and return TNC results.
    
    env: SkittlesEnv
    seed: random seed for reproducibility
    init_mean: np.array([angle, velocity]) starting mean for policy
    """
    np.random.seed(seed)
    #init_mean=[20, 2]
    
    # reset learner with given mean
    learner = SkittlesLearner(
        init_mean=init_mean,
        init_std=[10, .7],
        alpha=0.07,
        alpha_nu=0.07,
        alpha_phi=0.07,
        rwd_baseline_decay=0.99)
    actions = []
    
    # simulate trials
    for t in range(n_trials):
        action = learner.select_action()
        _, reward, _, _, info = env.step(action)
        learner.update(action, reward)   # your update rule
        actions.append(action)
    
    actions = np.array(actions)
    
    # compute TNC per block
    tnc = TNCCost(env)
    n_blocks = actions.shape[0] // block_size
    results = []
    for b in range(n_blocks):
        block_actions = actions[b*block_size:(b+1)*block_size]
        res = tnc.compute_all(block_actions)
        res["Block"] = b
        results.append(res)

    init_actions = actions[:block_size]
    last_actions = actions[-block_size:]
    
    return pd.DataFrame(results), actions, init_actions, last_actions


def plot_run(env, init_actions, last_actions, df_results, rep, init_mean):
    """
    Make two plots:
      1. Initial policy mean + first samples in action space
      2. TNC costs across blocks
    """
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # -----------------------------------
    # 1. Initial policy in execution space
    # -----------------------------------
    ax = axes[0]
    A_deg, V, R = env.compute_reward_grid(return_degrees = True)
    ax.pcolormesh(A_deg, V, R, shading='auto', cmap='gray', alpha=0.9)
    ax.scatter(np.rad2deg(init_actions[:60,0]), init_actions[:60,1], alpha=0.6, label="First block samples")
    ax.scatter(np.rad2deg(last_actions[:60,0]), last_actions[:60,1], alpha=0.6, label="Final block samples")
    ax.scatter(init_mean[0], init_mean[1], c="red", marker="x", s=100, label="Init mean")

    ax.set_xlabel("Launch Angle (degrees)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"Initial policy (rep {rep})")
    ax.set_xlim(A_deg.min(), A_deg.max())
    ax.set_ylim(V.min(), V.max())
    ax.legend()

    # -----------------------------------
    # 2. TNC costs across practice
    # -----------------------------------
    ax = axes[1]
    ax.plot(df_results["Block"], df_results["T-Cost"], label="T-Cost")
    ax.plot(df_results["Block"], df_results["N-Cost"], label="N-Cost")
    ax.plot(df_results["Block"], df_results["C-Cost"], label="C-Cost")
    ax.set_xlabel("Block (60 trials)")
    ax.set_ylabel("Cost")
    ax.set_title("TNC-Cost evolution")
    ax.legend()

    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# Run multiple simulations
# -------------------------------------------------

# set up environment
env = SkittlesEnv(target=[1, .4])

n_reps = 10
all_runs = []

# define action space ranges (radians & rad/s)
angle_range = (10, 160)               # 0 to -180°
vel_range   = (2, 9)

for rep in range(n_reps):
    seed = rep
    init_angle = np.random.uniform(*angle_range)
    init_vel   = np.random.uniform(*vel_range)
    init_mean  = np.array([init_angle, init_vel])
    
    df, actions, init_actions, last_actions = run_experiment(env, seed, init_mean)
    df["Rep"] = rep
    actions = np.array(actions)  # or however you keep actions
    plot_run(env, init_actions, last_actions, df, rep, init_mean)

    all_runs.append(df)

all_results = pd.concat(all_runs, ignore_index=True)

# %% plot results
# Plot mean across repetitions
mean_results = all_results.groupby("Block")[["T-Cost","N-Cost","C-Cost"]].mean()
mean_results.plot(title="Average TNC-Cost across repetitions", figsize=(5,4))

plt.savefig("TNC_analysis.svg", format="svg", bbox_inches='tight')

# Or inspect individual runs
#for rep in range(n_reps):
#    subset = all_results[all_results["Rep"]==rep]
#    plt.plot(subset["Block"], subset["T-Cost"], alpha=0.3)
#    plt.title("T-Cost across repetitions")
# %%
