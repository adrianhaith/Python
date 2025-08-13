#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:54:08 2025

@author: adrianhaith
"""

# script to define environment and agent models for the skittles task
import numpy as np
    
# agent model - defines the policy and learning rules
class PGLearner:
    def __init__(self, init_mean=None, init_std=None, alpha_mu=0.01, alpha_nu=0.01, alpha_phi=0.01, rwd_baseline_decay=0.99):
        self.alpha_mu = alpha_mu
        self.alpha_nu = alpha_nu
        self.alpha_phi = alpha_phi

        # Defaults
        if init_mean is None:
            init_mean = np.array([0.5, 0.5])
        if init_std is None:
            init_std = np.array([.1, .1]) 

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
            _, reward, _, _ = env.step(action)
            rewards.append(reward)
        self.rwd_baseline = np.mean(rewards)
    
    def select_action(self):
        cov = self._covariance_norm()
        a_norm = np.random.multivariate_normal(self.mu_norm, cov)
        self._last_action_norm = a_norm
        a_real = self._from_normalized(a_norm)
        return a_real  # convert angle to un-normalized action

    def update(self, action_real, reward):
        # Convert back to un-normalized coordinates to feed back to the environment
        delta_real = action_real - self.init_mean
        delta_norm = delta_real / self.init_std

        # --- Mean update (in normalized space) ---
        cov = self._covariance_norm()
        grad_logp_mu = np.linalg.inv(cov) @ (delta_norm - self.mu_norm)
        self.mu_norm += self.alpha_mu * (reward - self.rwd_baseline) * grad_logp_mu

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
    def mean(self):
        return self._from_normalized(self.mu_norm)
    
    # attribute to access the un-normalized covariance matrix
    @property
    def cov(self):
        covariance_norm = self._covariance_norm()
        scaling = np.array([[self.init_std[0]**2,0], [0, self.init_std[1]**2]])
        return scaling * covariance_norm




# define a simple version of the PGLearner class that neglects the normalization
class PGLearnerSimple:
    def __init__(self, init_mean=None, init_std=None, alpha_mu=0.01, alpha_nu=0.01, alpha_phi=0.01, rwd_baseline_decay=0.99):
        self.alpha_mu = alpha_mu
        self.alpha_nu = alpha_nu
        self.alpha_phi = alpha_phi

        # Defaults
        if init_mean is None:
            init_mean = np.array([0.5, 0.5])
        if init_std is None:
            init_std = np.array([.1, .1]) 

        self.init_mean = np.array(init_mean)
        self.init_std = np.array(init_std)

        # Learnable parameters (in normalized space)
        self.mu = init_mean                   # policy mean
        self.nu = np.log([init_std[0]**2, init_std[1]**2])            # log-eigenvalues of policy covariance matrix
        self.phi = 0.0                        # orientation parameter for covariance matrix
        self.Q = self._rotation_matrix(self.phi)

        self.rwd_baseline = 0.0
        self.rwd_baseline_decay = rwd_baseline_decay

    def _rotation_matrix(self, phi):
        return np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi),  np.cos(phi)]
        ])

    def _covariance(self):
        Lambda = np.diag(np.exp(self.nu))
        return self.Q @ Lambda @ self.Q.T

    def initialize_rwd_baseline(self, env, n_samples=100):
        rewards = []
        for _ in range(n_samples):
            env.reset()
            action = self.select_action()
            _, reward, _, _ = env.step(action)
            rewards.append(reward)
        self.rwd_baseline = np.mean(rewards)
    
    def select_action(self):
        cov = self._covariance()
        action = np.random.multivariate_normal(self.mu, cov)
        self._last_action = action
        return action  # convert angle to un-normalized action

    def update(self, action, reward):
        cov = self._covariance()
        grad_logp_mu = np.linalg.inv(cov) @ (action - self.mu)
        self.mu += self.alpha_mu * (reward - self.rwd_baseline) * grad_logp_mu

        # --- ν update ---
        z = self.Q.T @ (action - self.mu)
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
    def mean(self):
        return self.mu
    
    # attribute to access the un-normalized covariance matrix
    @property
    def cov(self):
        return self._covariance()
