#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:21:20 2025

@author: adrianhaith
"""

# define models of the environment and the learner

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

# define the skittles environment class

class CursorControlEnv(gym.Env):
    def __init__(self, radius=0.12):
        super().__init__()
        self.radius = radius
        self.target_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        self.action_space = Box(low=-2*radius, high=2*radius, shape=(4,), dtype=np.float32)  # Lx, Ly, Rx, Ry
        self.observation_space = Discrete(len(self.target_angles))  # Target angle index
        self.state = None
        self.target_pos = None

    def reset(self):
        idx = np.random.randint(len(self.target_angles))
        angle = self.target_angles[idx]
        self.state = angle
        self.target_pos = self.radius * np.array([np.cos(angle), np.sin(angle)])
        return angle  # angle is the state

    def step(self, action):
        # action = [Lx, Ly, Rx, Ry]
        Lx, Ly, Rx, Ry = action
        cursor_x = Ly  # controlled by L_y
        cursor_y = Rx  # controlled by R_x
        cursor_pos = np.array([cursor_x, cursor_y])
        error = np.linalg.norm(cursor_pos - self.target_pos)
        reward = -error ** 2
        return self.state, reward, True, {}
    
from scipy.special import i0  # Bessel function of the first kind, order 0

# define the learner class to model RL-based learning of the task

class CursorControlLearner:
    def __init__(self, 
                 n_basis=36, 
                 kappa=5.0, 
                 alpha=0.01, 
                 alpha_nu=0.01, 
                 sigma=0.15,
                 init_nu=None, 
                 seed=None, 
                 baseline_decay=0.99,
                 #radius=1
                ):
        self.n_basis = n_basis
        self.kappa = kappa
        self.alpha = alpha
        self.alpha_nu = alpha_nu
        self.rng = np.random.default_rng(seed)

        self.basis_centers = np.linspace(0, 2 * np.pi, n_basis, endpoint=False)
        self.W = self.rng.normal(scale=sigma, size=(4, n_basis))

        if init_nu is None:
            init_nu = np.log([sigma, sigma, sigma, sigma])
        self.nu = np.array(init_nu)

        self.rwd_baseline = 0.0
        self.baseline_decay = baseline_decay

    def compute_basis(self, s):
        diffs = s - self.basis_centers
        unnormalized = np.exp(self.kappa * np.cos(diffs))
        normalization = 2 * np.pi * i0(self.kappa)
        return unnormalized / normalization  # shape (n_basis,)

    def sample_action(self, s):
        phi = self.compute_basis(s)            # (n_basis,)
        mu = self.W @ phi                      # (4,)
        sigma = np.exp(self.nu)                # (4,)
        action = self.rng.normal(mu, sigma)    # (4,)
        return action, mu, sigma, phi

    def log_prob(self, a, mu, sigma):
        return -0.5 * np.sum(((a - mu) / sigma) ** 2 + 2 * self.nu + np.log(2 * np.pi))

    def update(self, action, mu, sigma, phi, reward):
        # Subtract baseline
        adv = reward - self.rwd_baseline
        self.rwd_baseline = self.baseline_decay * self.rwd_baseline + (1 - self.baseline_decay) * reward

        delta = action - mu
        grad_W = np.outer(delta / (sigma**2), phi) * adv
        grad_nu = ((delta**2) / (sigma**2) - 1) * adv
        
        self.W += self.alpha * grad_W
        self.nu += self.alpha_nu * grad_nu

    def initialize_baseline(self, env, n_trials=100):
        rewards = []
        actions = []
        for _ in range(n_trials):
            s = env.reset()
            a, mu, sigma, phi = self.sample_action(s)
            _, r, _, _ = env.step(a)
            rewards.append(r)
            actions.append(a)

        self.rwd_baseline = np.mean(rewards)
        return np.mean(rewards)

    def evaluate_policy_over_angles(self, n_points=100):
        """
        Returns:
            angles: array of target angles (radians)
            means: array of shape (n_points, 4) with policy means at each angle
        """
        angles = np.linspace(0, 2 * np.pi, n_points)
        means = np.zeros((n_points, 4))

        for i, angle in enumerate(angles):
            phi = self.compute_basis(angle)
            mu = self.W @ phi  # (4,)
            means[i] = mu
            
        return angles, means
    