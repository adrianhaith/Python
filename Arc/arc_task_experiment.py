import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from arc_task_env import ArcTaskEnv, make_arc_subgoals
from traj_learner import TrajLearner
from wrist_model import WristLDS
from plotting import plot_arc_trials, jitter_and_average

def run_simulation(n_trials=1000, seed=0):
    np.random.seed(seed)

    #create environment
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
                                baseline_decay=.95,
                                epsilon=0.1)
    
    participant.initialize_baseline(arc_env)

    history = {
        'actions': np.zeros((n_trials,2,6)),
        'rewards': np.zeros(n_trials),
        'stds': np.zeros((n_trials, 12)),
        'trajectories': np.zeros((n_trials, NT, 2)),
        'radial_pos' : np.zeros((n_trials, NT))
    }
    
    for trial in range(n_trials):
        action = participant.sample_action()
        _, reward, _, info = arc_env.step(action)
        participant.update(action, reward)
        
        # update history
        history['rewards'][trial] = reward
        history['actions'][trial] = action
        trajectory = info['trajectory'][:,[0,5]] # position x-y trajectory
        history['trajectories'][trial] = trajectory # x position
        #history['trajectories'][trial,:,1] = info['trajectory'][:,5] # y position
        history['stds'][trial] = participant.init_std * np.sqrt(np.exp(participant.nu))

        # compute radial position
        radial_pos = np.sqrt((arc_env.radius-trajectory[:,0])**2 + trajectory[:,1]**2)
        history['radial_pos'][trial] = radial_pos

    #plot_arc_trials(arc_env, participant, n_trials=5)

    return history, NT
    
# helper function to bin data for cleaner plots
def bin_data(array, bin_size=50):
    """
    Returns average values of input array binned into bins of size block_size
    """
    n_bins = len(array) // bin_size
    trimmed = array[:n_bins * bin_size]  # drop incomplete final block
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    bin_centers = bin_size*(np.arange(n_bins)+.5)
    return binned, n_bins, bin_size, bin_centers


# %% run multiple simulations
n_runs = 100
n_trials = 1000
bin_size = 50

all_rewards = []
all_dir_errors = []

early_radial_dist = []
late_radial_dist = []

early_mean_trajs = []
late_mean_trajs = []

radial_pos_all_runs = []

time = []
for run in range(n_runs):
    np.random.seed(run)
    hist, NT = run_simulation(n_trials=n_trials, seed=run)
    rwd_binned, n_bins, _, bin_centers = bin_data(hist['rewards'], bin_size=bin_size)
    all_rewards.append(rwd_binned)

    rng_early = np.arange(0,199)
    rng_late = np.arange(799,999)

    radial_pos_all_runs.append(hist['radial_pos'])



    time = .001 * np.arange(NT) # time vector for plotting later

    print("run",{run})

    #fig, axs = plt.subplots(1, 1, figsize=(4, 2), sharex=True)
    #traj = hist['radial_pos']

    #axs.plot(time, np.mean(hist['radial_pos'][rng_early,:],axis=0), label="early")
    #axs.plot(time, np.mean(hist['radial_pos'][rng_late,:],axis=0), label="late")

# %%
fig, axs = plt.subplots(2, 1, figsize=(4, 4), sharex=True)

bin_size = 50

axs[0].plot(bin_centers, np.mean(all_rewards, axis=0), marker='o', label='Reward')
axs[0].set_ylabel("Reward")
axs[0].set_xlabel("Trials")

for i in range(12):
    std_binned, _, _, bin_centers = bin_data(hist['stds'][:,i], bin_size=bin_size)
    axs[1].plot(bin_centers, std_binned, marker='o')
axs[1].set_ylabel("Std Dev")
axs[1].set_xlabel("Trial")
axs[1].legend()

# %% plot mean radial distance across runs

for run in range(n_runs):
    early_mean_trajs.append(jitter_and_average(radial_pos_all_runs[run][:100], dt=0.001, jitter_ms=150))
    late_mean_trajs.append(jitter_and_average(radial_pos_all_runs[run][100:], dt=0.001, jitter_ms=150))

plt.plot(time, np.mean(early_mean_trajs, axis=0), label="early")
plt.plot(time, np.mean(late_mean_trajs, axis=0), label="late")
plt.xlabel("time")
plt.ylabel("radial position")
plt.ylim(.25-.08, .25+.08)
plt.legend()
