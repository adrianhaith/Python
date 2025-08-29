# first, import needed libraries
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from plotting import plot_policy_snapshot
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

from models import SkittlesEnv, SkittlesLearner

make_animation = False

# %% Simulate learning

# Set random number seed
np.random.seed(3)

participant = SkittlesLearner(
    init_mean=[110, 2],
    init_std=[10, .75],
    alpha=0.05,
    alpha_nu=0.05,
    alpha_phi=0.05,
    rwd_baseline_decay=0.99
)
env = SkittlesEnv()

ax, out = env.plot_sample_trajectories(n_samples=1, seed=5)
plt.savefig("skittles_top_down.svg", format="svg", bbox_inches='tight')
