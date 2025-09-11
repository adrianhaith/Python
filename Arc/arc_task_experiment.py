import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from arc_task_env import ArcTaskEnv, make_arc_subgoals
from traj_learner import TrajLearner
from wrist_model import WristLDS
from plotting import plot_arc_trials

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
                                baseline_decay=.99,
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

    plot_arc_trials(arc_env, participant, n_trials=5)

    return history, NT
    
# helper function to bin data for cleaner plots
def bin_data(array, bin_size=50):
    """
    Returns average values of input array binned into bins of size block_size
    """
    n_bins = len(array) // bin_size
    trimmed = array[:n_bins * bin_size]  # drop incomplete final block
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    return binned, n_bins, bin_size


# %% run multiple simulations
n_runs = 5
n_trials = 1000
bin_size = 50

all_rewards = []
all_dir_errors = []

initial_radial_dist = []
late_radial_dist = []

time = []
for run in range(n_runs):
    np.random.seed(run)
    hist, NT = run_simulation(n_trials=n_trials, seed=run)
    rwd_binned, n_bins, _ = bin_data(hist['rewards'], bin_size=bin_size)
    all_rewards.append(rwd_binned)

    rng_early = np.arange(0,199)
    rng_late = np.arange(799,999)
    initial_radial_dist.append(np.mean(hist['radial_pos'][rng_early,:], axis=0))
    late_radial_dist.append(np.mean(hist['radial_pos'][rng_late,:], axis=0))

    time = .001 * np.arange(NT) # time vector for plotting later

    print("run",{run})

    fig, axs = plt.subplots(1, 1, figsize=(4, 8), sharex=True)
    traj = hist['radial_pos']

    axs.plot(time, np.mean(hist['radial_pos'][rng_early,:],axis=0), label="early")
    axs.plot(time, np.mean(hist['radial_pos'][rng_late,:],axis=0), label="late")

# %%




