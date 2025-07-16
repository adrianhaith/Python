#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:24:30 2025

@author: adrianhaith
"""
# define functions to help visualize the bimanual  task
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# function to visualize cursor behavior for a single trial
def plot_task_snapshot(target_angle, action, radius=0.12, n_targets=8, cmap=plt.cm.hsv):
    """
    Visualizes the target ring and the participant's endpoint on a single trial.

    Arguments:
        target_angle: angle (rad) of current trial's target
        action: (4,) array, where Ly = x and Rx = y
        radius: radius of target ring (in meters)
        n_targets: total number of possible targets (e.g. 8 or 16)
        cmap: matplotlib colormap to use (cyclic recommended)
    """
    # Compute canonical target positions
    canonical_angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)
    target_locs = np.stack([np.cos(canonical_angles), np.sin(canonical_angles)], axis=1) * radius

    def angle_to_color(angle):
        return cmap((angle % (2 * np.pi)) / (2 * np.pi))

    # Cursor position for this trial
    cursor_x = action[1]  # Ly
    cursor_y = action[2]  # Rx

    fig, ax = plt.subplots(figsize=(3, 3))

    # Plot central start position
    ax.plot(0, 0, 'ko', markersize=4)

    # Plot all targets
    for angle, (x, y) in zip(canonical_angles, target_locs):
        ax.plot(x, y, 'o', color=angle_to_color(angle), markersize=6)

    # Plot true target for this trial
    target_xy = radius * np.array([np.cos(target_angle), np.sin(target_angle)])
    ax.plot(*target_xy, 'o', color=angle_to_color(target_angle), markersize=10, markeredgecolor='black', label="Target")

    # Plot endpoint
    ax.plot(cursor_x, cursor_y, 'o', color=angle_to_color(target_angle), markersize=4, label="Cursor")

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    
def plot_learning_progress(actions, target_angles, radius=1, cmap=plt.cm.hsv):
    n_trials = len(actions)
    thirds = np.array_split(np.arange(n_trials), 3)

    # Use canonical target ring to plot context
    canonical_angles = np.linspace(0, 2 * np.pi, len(np.unique(target_angles)), endpoint=False)
    target_locs = np.stack([np.cos(canonical_angles), np.sin(canonical_angles)], axis=1) * radius

    def angle_to_color(angle):
        return cmap((angle % (2 * np.pi)) / (2 * np.pi))

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    titles = ['Early Trials', 'Middle Trials', 'Late Trials']

    for i, trial_indices in enumerate(thirds):
        ax = axs[i]
        ax.set_title(titles[i])

        # Plot center
        ax.plot(0, 0, 'ko', markersize=3)

        # Plot canonical targets
        for angle, (x, y) in zip(canonical_angles, target_locs):
            ax.plot(x, y, 'o', color=angle_to_color(angle), markersize=6)

        # Plot endpoints
        for t in trial_indices:
            angle = target_angles[t]
            action = actions[t]
            endpoint = [action[1], action[2]]  # Ly, Rx
            ax.plot(*endpoint, 'o', color=angle_to_color(angle), markersize=2, alpha=0.6)

        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.show()    
    
    
def make_animation(actions, target_angles, radius=1, cmap=plt.cm.hsv,
                     window=100, step=20, save_path=None):
    """
    Creates an animation showing learning progression using a sliding window.

    Parameters:
        actions: (n_trials, 4) array of agent actions
        target_angles: (n_trials,) array of angles in radians
        radius: radius of the target ring
        cmap: cyclic colormap (e.g., plt.cm.hsv)
        window: number of trials to show per frame
        step: number of trials between frames
        save_path: if provided, path to save animation as .mp4
    """
    n_trials = len(actions)
    canonical_angles = np.linspace(0, 2 * np.pi, len(np.unique(target_angles)), endpoint=False)
    target_locs = np.stack([np.cos(canonical_angles), np.sin(canonical_angles)], axis=1) * radius

    def angle_to_color(angle):
        return cmap((angle % (2 * np.pi)) / (2 * np.pi))

    fig, ax = plt.subplots(figsize=(3, 3))

    def init():
        ax.clear()
        ax.set_aspect('equal')
        ax.axis('off')

    def update(frame_idx):
        ax.clear()
        ax.set_aspect('equal')
        ax.axis('off')
        start = frame_idx * step
        end = start + window
        end = min(end, n_trials)

        # Plot center
        ax.plot(0, 0, 'ko', markersize=3)

        # Plot canonical targets
        for angle, (x, y) in zip(canonical_angles, target_locs):
            ax.plot(x, y, 'o', color=angle_to_color(angle), markersize=6)

        # Plot endpoints for current window
        for t in range(start, end):
            angle = target_angles[t]
            endpoint = [actions[t][1], actions[t][2]]  # Ly, Rx
            ax.plot(*endpoint, 'o', color=angle_to_color(angle), markersize=2, alpha=0.6)

        ax.set_title(f"Trials {start+1}–{end}")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

    n_frames = max(1, (n_trials - window) // step + 1)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=25)

    if save_path:
        anim.save(save_path, writer='ffmpeg', dpi=150)
    else:
        plt.close(fig)
        return anim    
    
# now, define a function to plot the policy pre vs post a given trial

def plot_policy_update(W_pre, W_post, nu, target_angle, learner, 
                       action=None, r=0.12, n_points=200):
    """
    Visualizes the policy before and after an update, with target and cursor endpoint.
    
    Arguments:
        W_pre, W_post: (4, n_basis) weight matrices before/after update
        nu: log std devs (length-4)
        target_angle: angle (rad) for this trial
        learner: CursorControlLearner object (to access basis function logic)
        action: (4,) array representing the sampled action (optional)
        r: radius of the target ring
        n_points: number of target angles to evaluate
    """
    angles = np.linspace(0, 2 * np.pi, n_points)
    phi_matrix = np.array([learner.compute_basis(s) for s in angles])  # (n_points, n_basis)

    mu_pre = phi_matrix @ W_pre.T    # (n_points, 4)
    mu_post = phi_matrix @ W_post.T  # (n_points, 4)

    ideal_Ly = r * np.cos(angles)
    ideal_Rx = r * np.sin(angles)

    # Target and endpoint coordinates
    target_xy = np.array([r * np.cos(target_angle), r * np.sin(target_angle)])
    if action is not None:
        cursor_xy = np.array([action[1], action[2]])  # Ly = x, Rx = y

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

    # Ly = cursor x control
    axs[0].plot(angles, mu_pre[:, 1], label='Pre-update', color='gray')
    axs[0].plot(angles, mu_post[:, 1], label='Post-update', color='blue')
    axs[0].plot(angles, ideal_Ly, '--', label='Ideal', color='black')
    axs[0].axvline(target_angle, color='red', linestyle=':', label='Target angle')
    axs[0].scatter([target_angle], [target_xy[0]], color='red', label='Target X', zorder=5)
    if action is not None:
        axs[0].scatter([target_angle], [cursor_xy[0]], color='green', label='Cursor X', zorder=5)
    axs[0].set_title("Left Hand Y (Ly)")
    axs[0].set_ylabel("Ly displacement (cursor x)")
    axs[0].grid(True)
    axs[0].legend()

    # Rx = cursor y control
    axs[1].plot(angles, mu_pre[:, 2], label='Pre-update', color='gray')
    axs[1].plot(angles, mu_post[:, 2], label='Post-update', color='blue')
    axs[1].plot(angles, ideal_Rx, '--', label='Ideal', color='black')
    axs[1].axvline(target_angle, color='red', linestyle=':')
    axs[1].scatter([target_angle], [target_xy[1]], color='red', label='Target Y', zorder=5)
    if action is not None:
        axs[1].scatter([target_angle], [cursor_xy[1]], color='green', label='Cursor Y', zorder=5)
    axs[1].set_title("Right Hand X (Rx)")
    axs[1].set_ylabel("Rx displacement (cursor y)")
    axs[1].grid(True)
    axs[1].legend()

    for ax in axs:
        ax.set_xlabel("Target angle (radians)")
        ax.set_xlim(0, 2 * np.pi)

    plt.suptitle("Policy Before and After Update + Target and Endpoint")
    plt.tight_layout()
    plt.show()

    