#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:54:08 2025

@author: adrianhaith
"""

# script to define environment and agent models for the skittles task
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# environment model - defines the skittles task, action space, reward etc.
class SkittlesEnv(gym.Env):
    def __init__(self, dt=0.01, T=2.0):
        super().__init__()

        # Physical constants
        self.k = 1.0     # spring stiffness
        self.m = 0.1     # mass
        self.l = 0.2     # paddle length
        self.xp = 0.0    # paddle pivot x
        self.yp = -1.5   # paddle pivot y

        self.target = np.array([0.8, 0.8], dtype=np.float32)

        # Action: [angle (rad), velocity (m/s)]
        self.action_space = spaces.Box(
        low=np.array([np.pi, 0.0], dtype=np.float32),
        high=np.array([2 * np.pi, 10.0], dtype=np.float32),
            dtype=np.float32
        )

        # Dummy observation
        self.observation_space = spaces.Box(
            low=np.array([0.0]), high=np.array([1.0]), shape=(1,), dtype=np.float32
        )

        # Time vector for simulation
        self.dt = dt
        self.T = T
        self.t = np.arange(0, T + dt, dt)
        self.omega = np.sqrt(self.k / self.m)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        angle, v = action

        # Initial release position
        xr = self.xp - self.l * np.cos(angle)
        yr = self.yp + self.l * np.sin(angle)

        # Initial velocity
        vx = v * np.sin(angle)
        vy = v * np.cos(angle)

        # Trajectory under central spring dynamics
        x_t = xr * np.cos(self.omega * self.t) + (vx / self.omega) * np.sin(self.omega * self.t)
        y_t = yr * np.cos(self.omega * self.t) + (vy / self.omega) * np.sin(self.omega * self.t)

        # Distance to target at each time step
        dx = x_t - self.target[0]
        dy = y_t - self.target[1]
        distances = np.sqrt(dx**2 + dy**2)

        # Reward: negative of the minimum distance
        reward = -np.min(distances)

        # Store full kinematic information in info
        info = {
            "trajectory": np.stack([x_t, y_t], axis=1),
            "min_distance": np.min(distances),
            "release_point": (xr, yr),
            "target": self.target
        }

        return np.array([0.0], dtype=np.float32), reward, True, False, info
    
    def compute_reward_grid(self, angle_range=None, velocity_range=None, resolution=300, return_degrees=False):
        """
        Computes a reward heatmap over the environment's action space.
    
        Parameters:
            angle_range (tuple): (min_angle, max_angle) in radians. Defaults to action_space angle range.
            velocity_range (tuple): (min_velocity, max_velocity). Defaults to action_space velocity range.
            resolution (int): number of points per axis.
            return_degrees (bool): if True, converts angle axis to degrees.
    
        Returns:
            A, V: meshgrid of angles and velocities
            R: reward matrix
        """
        # Use environment's action space bounds by default
        if angle_range is None:
            angle_range = (self.action_space.low[0], self.action_space.high[0])
        if velocity_range is None:
            velocity_range = (self.action_space.low[1], self.action_space.high[1])
    
        angles = np.linspace(angle_range[0], angle_range[1], resolution)
        velocities = np.linspace(velocity_range[0], velocity_range[1], resolution)
        A, V = np.meshgrid(angles, velocities)
        R = np.zeros_like(A)
    
        for i in range(resolution):
            for j in range(resolution):
                action = np.array([A[i, j], V[i, j]])
                _, reward, _, _, _ = self.step(action)
                R[i, j] = reward
    
        if return_degrees:
            A = np.rad2deg(A)
    
        return A, V, R
    
    
# agent model - defines the policy and learning rules
class SkittlesLearner:
    def __init__(self, init_mean=None, init_std=None, alpha=0.01, alpha_nu=0.01, alpha_phi=0.01, rwd_baseline_decay=0.99):
        self.alpha = alpha
        self.alpha_nu = alpha_nu
        self.alpha_phi = alpha_phi

        # Defaults
        if init_mean is None:
            init_mean = np.array([200.0, 2.0])
        if init_std is None:
            init_std = np.array([20.0, 0.5])  # 20° and 0.5 m/s

        self.init_mean = np.array(init_mean)
        self.init_std = np.array(init_std)

        # Learnable parameters (in normalized space)
        self.mu_norm = np.zeros(2)                      # normalized mean = 0
        self.nu = np.zeros(2)                           # log-eigenvalues in normalized units
        self.phi = 0.0
        self.Q = self._rotation_matrix(self.phi)

        self.rwd_baseline = 0.0
        self.rwd_baseline_decay = rwd_baseline_decay

    def _rotation_matrix(self, phi):
        return np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi),  np.cos(phi)]
        ])

    def _covariance_norm(self):
        Lambda = np.diag(np.exp(self.nu))
        return self.Q @ Lambda @ self.Q.T

    def _to_normalized(self, action_real):
        return (action_real - self.init_mean) / self.init_std

    def _from_normalized(self, action_norm):
        return self.init_mean + action_norm * self.init_std

    def initialize_rwd_baseline(self, env, n_samples=100):
        rewards = []
        for _ in range(n_samples):
            env.reset()
            action = self.select_action()
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)
        self.rwd_baseline = np.mean(rewards)
    
    def select_action(self):
        cov = self._covariance_norm()
        a_norm = np.random.multivariate_normal(self.mu_norm, cov)
        self._last_action_norm = a_norm
        a_real = self._from_normalized(a_norm)
        return np.array([np.deg2rad(a_real[0]), a_real[1]])  # convert angle to radians for env

    def update(self, action_real_rad, reward):
        # Convert back to real degrees for consistency with init_mean/init_std
        action_deg = np.array([np.rad2deg(action_real_rad[0]), action_real_rad[1]])
        delta_real = action_deg - self.init_mean
        delta_norm = delta_real / self.init_std

        # --- Mean update (in normalized space) ---
        cov = self._covariance_norm()
        grad_logp_mu = np.linalg.inv(cov) @ (delta_norm - self.mu_norm)
        self.mu_norm += self.alpha * (reward - self.rwd_baseline) * grad_logp_mu

        # --- ν update ---
        z = self.Q.T @ (delta_norm - self.mu_norm)
        lambdas = np.exp(self.nu)
        grad_logp_nu = 0.5 * (-1 + (z ** 2) / lambdas)
        self.nu += self.alpha_nu * (reward - self.rwd_baseline) * grad_logp_nu

        # --- φ update ---
        Lambda_inv = np.diag(1.0 / lambdas)
        grad_logp_dQ = - self.Q.T @ np.outer(z, z) @ Lambda_inv
        dQ_dphi = np.array([
            [-np.sin(self.phi), -np.cos(self.phi)],
            [ np.cos(self.phi), -np.sin(self.phi)]
        ])
        grad_logp_phi = np.trace(grad_logp_dQ.T @ dQ_dphi)
        self.phi += self.alpha_phi * (reward - self.rwd_baseline) * grad_logp_phi
        self.Q = self._rotation_matrix(self.phi)

        # --- Update the reward baseline
        self.rwd_baseline = self.rwd_baseline_decay * self.rwd_baseline + (1 - self.rwd_baseline_decay) * reward

    # attribute to access the un-normalized mean
    @property
    def mu(self):
        return self._from_normalized(self.mu_norm)
    
    # attribute to access the un-normalized covariance matrix
    @property
    def covariance(self):
        cov_norm = self._covariance_norm()
        scaling_mat = np.diag(self.init_std)
        return scaling_mat @ cov_norm @ scaling_mat