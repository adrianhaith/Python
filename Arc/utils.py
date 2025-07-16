#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:10:55 2025

@author: adrianhaith
"""

import numpy as np
import matplotlib.pyplot as plt

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
    subgoals = learner.get_policy_mean()
    ax.plot(np.hstack(([0], subgoals[0, :])) , np.hstack(([0], subgoals[1,:])), linestyle='-', color='red', marker='o', linewidth=.5, markersize=1, alpha=.5)

    
    # Simulate and plot trials
    for _ in range(n_trials):
        subgoals = learner.sample_action()
        _, _, _, info = env.step(subgoals)
        #cursor = info["cursor"]
        traj = info["trajectory"]
        ax.plot(traj[:, 0], traj[:, 5], color=color, alpha=1)
        
    ax.set_aspect('equal')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    #ax.set_xlim(-0.3, 0.3)
    #ax.set_ylim(-0.1, 0.6)
    plt.tight_layout()
    plt.show()
    
def plot_arc_trials2(env, learner, n_trials=5, color='C0'):
    """
    Plot arc trajectories and corresponding velocity profiles from sampled subgoal sequences.

    Parameters:
        env: ArcTaskEnv instance
        learner: ArcLearner instance
        n_trials: number of trials to simulate
        color: matplotlib color
    """
    r = env.radius
    w = env.width
    center = env.center
    theta = np.linspace(-np.pi/2, np.pi/2, 300)

    arc_outer = center[:, None] + (r + w/2) * np.vstack((np.sin(theta), +np.cos(theta)))
    arc_inner = center[:, None] + (r - w/2) * np.vstack((np.sin(theta), +np.cos(theta)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    # -- Trajectory Plot --
    ax1.plot(arc_outer[0], arc_outer[1], 'k--', linewidth=1)
    ax1.plot(arc_inner[0], arc_inner[1], 'k--', linewidth=1)
    for point in [np.array([0, 0]), np.array([0.5, 0.0])]:
        circle = plt.Circle(point, w/2, color='gray', fill=False, linestyle='--')
        ax1.add_patch(circle)

    # -- Velocity Plot Setup --
    t_vec = np.arange(0, env.lds.T + env.dt, env.dt)

    for _ in range(n_trials):
        subgoals, flat_action, noise = learner.sample_action()
        _, _, _, info = env.step(subgoals)

        cursor = info["cursor"]
        traj = info["trajectory"]

        pos_x = traj[:,0]
        pos_y = traj[:,5]

        # Plot trajectory
        ax1.plot(pos_x, pos_y, color=color, alpha=0.7)

        # Extract velocities
        vel_x = traj[:, 1]
        vel_y = traj[:, 6]
        ax2.plot(t_vec, vel_x, label='vx', color='tab:blue', alpha=0.5)
        ax2.plot(t_vec, vel_y, label='vy', color='tab:orange', alpha=0.5)

    # -- Axes Formatting --
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Cursor Trajectories")
    #ax1.set_xlim(-0.3, 0.3)
    #ax1.set_ylim(-0.1, 0.6)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_title("Velocity Profiles")
    ax2.grid(True)

    # Add one legend (avoid clutter)
    ax2.plot([], [], label='vx', color='tab:blue')
    ax2.plot([], [], label='vy', color='tab:orange')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    
