#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:10:55 2025

@author: adrianhaith
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_arc_trials(env, learner, n_trials=5, color='C0', title="cursor trajectories"):
    """
    Simulate and plot several cursor trajectories using the current policy.

    Parameters:
        env: ArcTaskEnv instance
        learner: ArcLearner instance
        n_trials: number of trials to simulate
        color: matplotlib color for trajectories
    """
    # Arc parameters
    r = env.radius
    w = env.width
    center = env.center
    theta = np.linspace(-np.pi/2, np.pi/2, 300)


    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')

    # Plot arc boundaries
    arc_outer = center[:, None] + (r + w/2) * np.vstack((np.sin(theta), np.cos(theta)))
    arc_inner = center[:, None] + (r - w/2) * np.vstack((np.sin(theta), np.cos(theta)))
    ax.plot(arc_outer[0], arc_outer[1], 'k-', linewidth=1)
    ax.plot(arc_inner[0], arc_inner[1], 'k-', linewidth=1)

    # Plot start and end circles
    for point in [np.array([0, 0]), np.array([0.5, 0])]:
        circle = plt.Circle(point, w/2, color='gray', fill=False, linestyle='-')
        ax.add_patch(circle)


    # get and plot mean goal locations
    mean_subgoals = learner.get_policy_mean()
    ax.plot(np.hstack(([0], mean_subgoals[0, :])) , np.hstack(([0], mean_subgoals[1,:])), linestyle='-', color='red', marker='o', linewidth=2, markersize=5, alpha=1)

    
    # Simulate and plot trials
    for _ in range(n_trials):
        subgoals = learner.sample_action()
        _, _, _, info = env.step(subgoals)
        #cursor = info["cursor"]
        traj = info["trajectory"]
        ax.plot(traj[:, 0], traj[:, 5], color=color, alpha=1)
        
    # now plot ellipses around the subgoal points to illustate policy variance
    n_goals = learner.Ng
    stds = learner.init_std * np.sqrt(np.exp(learner.nu))
    std_x = stds[:n_goals]
    std_y = stds[n_goals:]

    for i in range(n_goals):
        mu_x = mean_subgoals[0, i]
        mu_y = mean_subgoals[1, i]
        width = 2 * std_x[i]
        height = 2 * std_y[i]
        ellipse = Ellipse((mu_x, mu_y), width=width, height=height,
                          edgecolor='red', facecolor='none', linewidth=1, linestyle='-', alpha=1)
        ax.add_patch(ellipse)


    ax.set_aspect('equal')
    ax.set_xlim([-0.1, 0.65])
    ax.set_ylim([-.05, .35])
    ax.set_axis_off()
    #ax.set_xlim(-0.3, 0.3)
    #ax.set_ylim(-0.1, 0.6)
    plt.tight_layout()
    #plt.show()

class ArcVisualizer:
    def __init__(self, env, learner, history):
        self.env = env
        self.learner = learner
        self.history = history
        self.Ng = learner.Ng
        self.r = env.radius
        self.w = env.width
        self.center = env.center
        self.theta = np.linspace(-np.pi/2, np.pi/2, 300)

    def plot_trials(self, start_trial, n_trials=5, color='C0', title="Cursor trajectories (from history)"):
        
        trials = range(start_trial, start_trial+n_trials)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal', adjustable='box')

        # --- Plot arc geometry ---
        arc_outer = self.center[:, None] + (self.r + self.w/2) * np.vstack((np.sin(self.theta), np.cos(self.theta)))
        arc_inner = self.center[:, None] + (self.r - self.w/2) * np.vstack((np.sin(self.theta), np.cos(self.theta)))
        ax.plot(arc_outer[0], arc_outer[1], 'k-', linewidth=1)
        ax.plot(arc_inner[0], arc_inner[1], 'k-', linewidth=1)
        for point in [np.array([0, 0]), np.array([0.5, 0])]:
            circle = plt.Circle(point, self.w/2, color='gray', fill=False, linestyle='-')
            ax.add_patch(circle)

        # --- Plot each trajectory ---
        for t in trials:
            traj = self.history['trajectories'][t]  # shape (NT, 2)
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=1.0)

            # --- Plot subgoals for this trial---
            #subgoals = self.history['actions'][t]
            mean_subgoals = self.history['means'][t]
            mean_subgoals_x = mean_subgoals[:self.Ng]
            mean_subgoals_y = mean_subgoals[self.Ng:]
            ax.plot(mean_subgoals_x, mean_subgoals_y, linestyle='-', color='red', marker='o', linewidth=2, markersize=5, alpha=1)

            # --- Plot ellipses for final trial's stds ---
            stds = self.history['stds'][t]
            std_x = stds[:self.Ng]
            std_y = stds[self.Ng:]

            for i in range(self.Ng):
                mu_x = mean_subgoals_x[i]
                mu_y = mean_subgoals_y[i]
                width = 2 * std_x[i]
                height = 2 * std_y[i]
                ellipse = Ellipse((mu_x, mu_y), width=width, height=height,
                                    edgecolor='red', facecolor='none', linewidth=1.5, linestyle='--', alpha=0.6)
                ax.add_patch(ellipse)

        # --- Final formatting ---
        ax.set_xlim([-0.1, 0.65])
        ax.set_ylim([-.05, .35])
        ax.set_title(title)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

# function to jitter different movements before taking average trajectories, to avoid arficially in-phase submovements
def jitter_and_average(radial_pos_trials, dt, jitter_ms=25, n_avg_trials=100):
    """
    Applies random temporal jitter to trial data and computes average radial trajectory.

    Parameters:
        radial_pos_trials: (n_trials, n_timesteps) array
        dt: timestep in seconds
        jitter_ms: max jitter to apply (symmetric) in milliseconds
        n_avg_trials: number of trials to include in average (e.g. 100)

    Returns:
        mean_radial: (n_timesteps,) array with nan-robust average
    """
    n_trials, n_timesteps = radial_pos_trials.shape
    jitter_steps = int(jitter_ms / 1000 / dt)

    padded = np.full((n_avg_trials, n_timesteps + 2 * jitter_steps), np.nan)

    for i in range(n_avg_trials):
        shift = np.random.randint(-jitter_steps, jitter_steps + 1)
        if shift >= 0:
            padded[i, jitter_steps + shift : jitter_steps + shift + n_timesteps] = radial_pos_trials[i]
        else:
            padded[i, jitter_steps + shift : jitter_steps + shift + n_timesteps] = radial_pos_trials[i]

    mean_radial = np.nanmean(padded, axis=0)

    # Return center segment of same original length
    return mean_radial[jitter_steps : jitter_steps + n_timesteps]