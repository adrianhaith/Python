#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:21:20 2025

@author: adrianhaith
"""

# define models of the environment and the learner

import numpy as np

# define the skittles environment class

class CursorControlEnv:
    def __init__(self, radius=0.12, discrete_targs=True):
        self.radius = radius
        self.target_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        u_range = (-2*radius, 2*radius)
        self.action_space = np.array([u_range, u_range, u_range, u_range])
        #self.action_space = Box(low=-2*radius, high=2*radius, shape=(4,), dtype=np.float32)  # Lx, Ly, Rx, Ry
        self.observation_space = np.array([0.0, 2*np.pi])  # Target angle index
        self.state = None
        self.target_pos = None
        self.discrete = discrete_targs

    def reset(self):
        if(self.discrete):
            idx = np.random.randint(len(self.target_angles)) # discrete selection
            angle = self.target_angles[idx]                  #. 
        else:
            angle = np.random.uniform(0, 2 * np.pi)

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
        reward = -np.linalg.norm(error) ** 2

        # Compute directional error
        targ_ang = np.atan2(self.target_pos[1],self.target_pos[0])
        reach_ang = np.atan2(cursor_pos[1],cursor_pos[0])
        angular_error = targ_ang - reach_ang
        abs_dir_error = np.abs(angular_error)

        # alternative calculation of directional error
        vec_target = self.target_pos
        vec_actual = cursor_pos

        dot = np.dot(vec_target, vec_actual)
        norm_prod = np.linalg.norm(vec_target) * np.linalg.norm(vec_actual) + 1e-8  # avoid 0/0
        cos_theta = np.clip(dot / norm_prod, -1.0, 1.0)
        angular_error = np.arccos(cos_theta)  # radians
        abs_dir_error = np.abs(angular_error)

        info = {
            'abs_directional_error': abs_dir_error,
            'cursor_pos': cursor_pos.copy(),
            'target_pos': self.target_pos.copy()
        }

        return self.state, reward, True, info
    
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
                 radius=.12,
                 epsilon=0.3 # clipping threshold for PPO-like stabilization
                ):
        self.n_basis = n_basis
        self.kappa = kappa
        self.alpha = alpha
        self.alpha_nu = alpha_nu
        self.rng = np.random.default_rng(seed)
        self.radius = radius

        self.basis_centers = np.linspace(0, 2 * np.pi, n_basis, endpoint=False)
        self.W = self.rng.normal(scale=self.radius/2, size=(4, n_basis))

        if init_nu is None:
            init_nu = np.log([sigma, sigma, sigma, sigma])
        self.nu = np.array(init_nu)

        self.V = np.zeros(self.n_basis)
        self.baseline_decay = baseline_decay

        self.epsilon = epsilon

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
        b = self.V @ phi
        adv = reward - b
        
        # Value-function update: weight the 
        self.V += (1 - self.baseline_decay) * adv * phi  # update value function by gradient descent

        # W-update
        delta = action - mu
        grad_W = np.outer(delta / (sigma**2), phi) * adv

        # clip updates according to PPO-based upper bound on change in mean
        for i in range(4):
            # Compute proposed update for this row of W
            delta_w_i = self.alpha * grad_W[i]  # shape (n_basis,)
            
            # Compute the corresponding change in mean action
            delta_mu_i = delta_w_i @ phi        # scalar
            
            # PPO-style bound
            max_step = self.epsilon * (sigma[i]**2) / abs(action[i] - mu[i] + 1e-8)  # add epsilon for stability

            if abs(delta_mu_i) > max_step:
                scale = max_step / abs(delta_mu_i)
                delta_w_i *= scale  # scale the update to satisfy constraint

            self.W[i] += delta_w_i

        grad_nu = ((delta**2) / (sigma**2) - 1) * adv
        self.nu += self.alpha_nu * grad_nu

    def initialize_baseline(self, env, n_trials=100):
        activations = [] # basis activations across trials
        rewards = []
        actions = []
        states = []
        for t in range(n_trials):
            s = env.reset()
            a, mu, sigma, phi = self.sample_action(s)
            _, r, _, _ = env.step(a)
            rewards.append(r)
            actions.append(a)
            activations.append(phi)
            states.append(s)

        Phi = np.stack(activations)
        R = np.array(rewards)
        self.V = np.linalg.lstsq(Phi, R, rcond=None)[0]
        


        return states, rewards
        

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
    