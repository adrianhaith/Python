import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_toy2d_reward_landscape(env, learner=None, actions=None, resolution=300, ax=None, RPE=False, policycolor='yellow',outline=False):
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
        #cmap = 'RdBu' # or 'coolwarm', 'RdBu'
        cmap = 'coolwarm'
        vlim = np.abs(R).max()
        heatmap = ax.pcolormesh(U1, U2, R, shading='auto', cmap=cmap, vmin=-vlim, vmax=vlim)
    else:
        cmap = 'gray' #'Greens' #'YlGn'
        heatmap = ax.pcolormesh(U1, U2, R, shading='auto', cmap=cmap, alpha=0.9)

    ax.set_xlim(env.u1_range)
    ax.set_ylim(env.u2_range)
    ax.set_xlabel('u1')
    ax.set_ylabel('u2')
    ax.set_title('Reward Landscape')

    # Policy mean + covariance ellipse
    if learner is not None:
        mu = learner.mean
        cov = learner.cov
        plot_covariance_ellipse(mu, cov, ax, color=policycolor, outline=outline)
        if outline:
            ax.plot(mu[0], mu[1], 'kx', markersize=10, alpha=0.6,markeredgewidth=3)
        ax.plot(mu[0], mu[1], marker='x', markersize=8, color=policycolor, markeredgewidth=1)
        
            # Sampled actions
    if actions:
        actions = np.array(actions)
        if outline:
            ax.scatter(actions[:, 0], actions[:, 1], marker='o', color=policycolor, edgecolor='black', s=20, label='Sampled Actions', alpha=1)
        else:
            ax.scatter(actions[:, 0], actions[:, 1], marker='o', color=policycolor, s=20, label='Sampled Actions')


    ax.set_aspect('equal')
    ax.legend()
    return ax

def plot_covariance_ellipse(mean, cov, ax, n_std=1.0, color='yellow',outline=False):
    """Plot an ellipse representing the covariance matrix."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    chisq_val = 5.991  # from chi^2 with 2 degrees of freedom - appropriate for plotting a 95% confidence interval for the covariance ellipse
    width, height = 2 * np.sqrt(chisq_val * eigvals)
    if outline:
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, edgecolor='black', linewidth=3, alpha=1)
        ax.add_patch(ellipse)
    ellipse2 = Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, edgecolor=color, linewidth=1)
    ax.add_patch(ellipse2)

    ax.plot(mean[0], mean[1], 'kx', markersize=10, alpha=1, markeredgewidth=3) # black background marker 
    ax.plot(mean[0], mean[1], marker='x', markersize=8, color=color, markeredgewidth=1) 
    return ellipse, ellipse2

def plot_policy_snapshot(ax, mu, nu, phi, color, n_samples=25, show_ellipse=True, label=None):
    """
    Plot a policy snapshot (samples + optional ellipse) on the given axes.

    Parameters:
    - ax: matplotlib Axes
    - mu: mean of policy (2,)
    - nu: log-eigenvalues (2,)
    - phi: rotation angle (scalar)
    - color: color to use for this snapshot
    - n_samples: number of action samples to draw
    - show_ellipse: whether to plot the 1 SD ellipse
    - label: optional label for the scatter points
    """
    # Reconstruct covariance
    Lambda = np.diag(np.exp(nu))
    Q = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi),  np.cos(phi)]
    ])
    cov = Q @ Lambda @ Q.T

    # Sample actions
    samples = np.random.multivariate_normal(mu, cov, size=n_samples)
    ax.scatter(samples[:, 0], samples[:, 1], s=10, color=color, label=label)

    if show_ellipse:
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * np.sqrt(vals)
        ellipse = Ellipse(mu, width, height, angle=angle, edgecolor=color,
                          facecolor='none', lw=1)
        ax.add_patch(ellipse)