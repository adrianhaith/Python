import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_toy2d_reward_landscape(env, learner=None, actions=None, resolution=300, ax=None, RPE=False):
    """
    Plots the reward heatmap for the Toy2DEnv.
    
    Parameters:
        env: Toy2DEnv instance.
        learner: SkittlesLearner instance (for showing mean and covariance).
        actions: list of sampled actions (for overlay).
        resolution: heatmap resolution.
        ax: optional matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Reward grid
    U1, U2, R = env.get_reward_grid(resolution=resolution)
    if RPE and learner is not None:
        R = R - learner.rwd_baseline

    if RPE:
        cmap = 'RdBu' # or 'coolwarm', 'RdBu'
        vlim = np.abs(R).max()
        heatmap = ax.pcolormesh(U1, U2, R, shading='auto', cmap=cmap, vmin=-vlim, vmax=vlim)
    else:
        cmap = 'cividis'
        heatmap = ax.pcolormesh(U1, U2, R, shading='auto', cmap=cmap, alpha=0.9)

    ax.set_xlim(env.u1_range)
    ax.set_ylim(env.u2_range)
    ax.set_xlabel('u1')
    ax.set_ylabel('u2')
    ax.set_title('Reward Landscape')

    # Sampled actions
    if actions:
        actions = np.array(actions)
        ax.scatter(actions[:, 0], actions[:, 1], color='red', s=10, label='Sampled Actions')

    # Policy mean + covariance ellipse
    if learner is not None:
        mu = learner.mean
        cov = learner.cov
        plot_covariance_ellipse(mu, cov, ax, edgecolor='white', lw=2)
        ax.scatter(mu[0],mu[1], color='white', s=20)

    ax.set_aspect('equal')
    ax.legend()
    return ax

def plot_covariance_ellipse(mean, cov, ax, n_std=1.0, **kwargs):
    """Plot an ellipse representing the covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    chisq_val = 5.991  # from chi^2 with 2 degrees of freedom - appropriate for plotting a 95% confidence interval for the covariance ellipse
    width, height = 2 * np.sqrt(chisq_val * eigvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, **kwargs)
    ax.add_patch(ellipse)
