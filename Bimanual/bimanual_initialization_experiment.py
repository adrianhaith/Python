# script to compare model learning against human learning after being initialized to the same initial policy

import numpy as np
from scipy.io import loadmat
from numpy.linalg import lstsq
from models import CursorControlLearner
from models import CursorControlEnv
from utils import compute_von_mises_basis, wrap_to_pi, wrap_to_2pi, ridge_fit, bin_data, analyze_bias_variability_sliding, fit_human_policy
import matplotlib.pyplot as plt
from visualization import CursorLearningVisualizer




# function to simulate learning that is initialized to mimic a human suject's initial policy
def simulate_learning_for_subject(subj_id, seed, n_trials=2200, window=60, n_basis=16):
    # Load subject data
    mat = loadmat('human_data_120.mat')
    subj_data = mat['human_data'][0][subj_id]
    target_angles = wrap_to_2pi(np.pi/2 - subj_data['target_angles'].squeeze())
    hand_movements = subj_data['hand_movements'].T

    # Fit initial policy
    W_init, std_init, nu_init = fit_human_policy(target_angles, hand_movements, n_basis=n_basis)

    # Set up environment and agent with seed
    env = CursorControlEnv(radius=.12, motor_noise_std=.075, discrete_targs=False, seed=seed)
    alpha = 0.1
    learner = CursorControlLearner(alpha=alpha, alpha_nu=alpha, sigma=std_init, seed=seed,
                                   baseline_decay=0.95, kappa=5, epsilon=0.4)
    learner.W = W_init.T.copy() / std_init
    learner.nu = nu_init.copy()
    learner.initialize_baseline(env, n_trials=1000)

    # Run learning
    dir_errors = np.zeros(n_trials)
    target_dirs = np.zeros(n_trials)
    for t in range(n_trials):
        s = env.reset()
        a, mu, sigma, phi = learner.sample_action(s)
        _, reward, _, info = env.step(a)
        learner.update(a, mu, sigma, phi, reward)

        dir_errors[t] = info['directional_error']
        target_dirs[t] = s

    # Analyze bias/variance decomposition
    bias_vars, resid_vars, total_vars = analyze_bias_variability_sliding(
        target_dirs, theta_errors=dir_errors, window=window
    )
    return dir_errors, bias_vars, resid_vars, total_vars


# script to average multiple runs for a single subject
def analyze_subject_group_model(subj_id, n_repeats=10, seed_offset=0):
    all_bias = []
    all_resid = []
    all_total = []

    for i in range(n_repeats):
        seed = seed_offset + i
        _, bias, resid, total = simulate_learning_for_subject(subj_id, seed)
        all_bias.append(bias)
        all_resid.append(resid)
        all_total.append(total)

    return {
        'bias': np.mean(all_bias, axis=0),
        'resid': np.mean(all_resid, axis=0),
        'total': np.mean(all_total, axis=0),
    }


#---- main part of the script: iterate over subjects and run a batch of simulations per subject. Average learning curves
#------------------

n_subjects = 13
n_repeats = 5

model_group_bias = []
model_group_resid = []
model_group_total = []

human_group_bias = []
human_group_resid = []
human_group_total = []

included_subjects = [0,1,2,3,6,7,8,9,10,11,12]
for subj_id in included_subjects:
    # Simulated model learner
    sim_results = analyze_subject_group_model(subj_id, n_repeats=n_repeats)

    # Human data
    mat = loadmat('human_de_novo_errors.mat')
    subj = mat['subject_data'][0][subj_id]
    bias_human, resid_human, total_human = analyze_bias_variability_sliding(
        wrap_to_2pi(subj['target_dir'].squeeze()*180/np.pi),
        theta_errors=subj['directional_error'].squeeze(),
        window=60
    )

    # Save for group average
    model_group_bias.append(sim_results['bias'])
    model_group_resid.append(sim_results['resid'])
    model_group_total.append(sim_results['total'])

    human_group_bias.append(bias_human)
    human_group_resid.append(resid_human)
    human_group_total.append(total_human)

# Average across subjects
model_mean_bias = np.mean(model_group_bias, axis=0)
model_mean_resid = np.mean(model_group_resid, axis=0)
model_mean_total = np.mean(model_group_total, axis=0)
human_mean_bias = np.mean(human_group_bias, axis=0)
human_mean_resid = np.mean(human_group_resid, axis=0)
human_mean_total = np.mean(human_group_total, axis=0)



# # %% series 
# for subj in range(np.size(included_subjects)):
#     fig, axs = plt.subplots(1, 2, figsize=(8, 2), sharey=True)

#     axs[0].plot(human_group_bias[subj], color='blue')
#     axs[0].plot(human_group_resid[subj], color='red')
#     axs[0].plot(human_group_total[subj], color='black')
#     axs[0].set_title(f"Human Learners - Subj {subj+1}")

#     axs[1].plot(model_group_bias[subj], label='Model', color='blue',ls='-')
#     axs[1].plot(model_group_resid[subj], label='Model', color='red',ls='-')
#     axs[1].plot(model_group_total[subj], label='Model', color='black',ls='-')
#     axs[1].set_title(f"Policy-Gradient RL - Subj {subj+1}")

# %% Plot All subjects
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

axs[0].plot(human_mean_bias[1:], label='Human', color='blue')
axs[0].plot(human_mean_resid[1:], label='Human', color='red')
axs[0].plot(human_mean_total[1:], label='Human', color='black')
axs[0].set_title('Human Learners - All Subjs')
axs[0].grid(True)

axs[1].plot(model_mean_bias[1:], label='Model', color='blue',ls='-')
axs[1].plot(model_mean_resid[1:], label='Model', color='red',ls='-')
axs[1].plot(model_mean_total[1:], label='Model', color='black',ls='-')
axs[1].set_title('Policy-Gradient RL - All Subjs')

axs[1].grid(True)

plt.show()

