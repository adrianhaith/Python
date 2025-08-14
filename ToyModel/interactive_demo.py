# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button

from toy_env import Toy2DEnv
from agent import PGLearnerSimple

# --- Helper for plotting ---
def plot_covariance_ellipse(mean, cov, ax, n_std=1.0, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    chisq_val = 5.991  # from chi^2 with 2 degrees of freedom - appropriate for plotting a 95% confidence interval for the covariance ellipse
    width, height = 2 * np.sqrt(chisq_val * eigvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

# --- Setup ---
env = Toy2DEnv()
learner = PGLearnerSimple(init_mean=[0.2, 0.2], init_std=[0.06, 0.03], alpha_mu=0.005, alpha_nu=0.02, alpha_phi=0.02, rwd_baseline_decay=0.9)
learner.initialize_rwd_baseline(env)
actions = []
#action = learner.select_action()

U1, U2, R = env.get_reward_grid(resolution=200)

fig, (ax_main, ax_cbar) = plt.subplots(
    1, 2, figsize=(8, 6), gridspec_kw={"width_ratios": [20, 1]}
)
# Create a new axis for the button (normalized figure coordinates)
button_ax = fig.add_axes([0.3, 0.9, 0.25, 0.07])  # [left, bottom, width, height]
sample_button = Button(button_ax, 'Sample Action', color='lightgray', hovercolor='gray')

heatmap = ax_main.pcolormesh(U1, U2, R, shading='auto', cmap='viridis', alpha=0.9)
ax_main.set_xlim(0, 1)
ax_main.set_ylim(0, 0.7)
ax_main.set_xlabel('u1')
ax_main.set_ylabel('u2')
ax_main.set_title('Click to select actions and update policy')
scatter = ax_main.scatter([], [], color='red', s=20, label='Selected actions')

# %%
# Initial ellipse
ellipse = [plot_covariance_ellipse(learner.mean, learner.cov, ax_main, edgecolor='white', lw=2)]

ax_main.legend()
ax_main.set_aspect('equal')

# dot to show last action
last_action_dot = ax_main.plot(
    [learner.mean[0]], [learner.mean[1]], 'b.', markersize=10, markeredgecolor='black'
)

# policy mean
mean_dot = ax_main.plot(
    [learner.mean[0]], [learner.mean[1]], 'w.', markersize=10
)

# --- Reward baseline colorbar ---
norm = plt.Normalize(vmin=R.min(), vmax=R.max())
cbar = plt.colorbar(heatmap, cax=ax_cbar)
cbar.set_label("Reward")

# --- Add reward baseline marker
rwd_dot = ax_cbar.plot(
    [0.5], [learner.rwd_baseline], 'ro', markersize=8, markeredgecolor='black'
)[0]

rwd_latest_dot = ax_cbar.plot(
    [0.5], [0.0], 'bo', markersize=8, markeredgecolor='black'
)[0]



ax_cbar.set_ylim(R.min(), R.max())
ax_cbar.set_xlim(0, 1)
ax_cbar.set_xticks([])
ax_cbar.set_yticks([])


# %% --- Interactive update ---
def update_plot(action):
    # Evaluate reward and update learner
    _, reward, _, _ = env.step(action)
    learner.update(action, reward)

    # Update scatter
    scatter.set_offsets(actions)

    # Update ellipse and mean dot
    ellipse[0].remove()
    ellipse[0] = plot_covariance_ellipse(learner.mean, learner.cov, ax_main, edgecolor='white', lw=2)
    mean_dot[0].set_data([learner.mean[0]], [learner.mean[1]])

    # Update last_action
    last_action_dot[0].set_data([action[0]],[action[1]])

    # Update reward dot on color bar
    rwd_dot.set_ydata([learner.rwd_baseline])
    rwd_latest_dot.set_ydata([reward])

    fig.canvas.draw()


def onclick(event):
    if event.inaxes != ax_main:
        return

    action = np.array([event.xdata, event.ydata])
    actions.append(action)

    update_plot(action)

def on_sample(event):
    action = learner.select_action()
    actions.append(action)

    update_plot(action)

sample_button.on_clicked(on_sample)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
# %%