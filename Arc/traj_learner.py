#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 08:48:40 2025

@author: adrianhaith
"""

import numpy as np

class TrajLearner:
    def __init__(self, Ng, alpha=0.01, alpha_nu=0.01, init_goals=None, init_std=0.05, baseline_decay=0.99):
        """
        Parameters:
            Ng: number of subgoals
            alpha: learning rate
            init_arc: optional array of shape (2, Ng) giving initial mean subgoal locations
            init_std: initial standard deviation (shared or scalar)
        """
        self.Ng = Ng
        self.alpha = alpha
        self.alpha_nu = alpha_nu
        
        self.action_dim = 2 * Ng

        self.rwd_baseline = 0
        self.baseline_decay = baseline_decay

        if init_goals is not None:
            self.mean = init_goals.flatten()
        else:
            self.mean = np.zeros(self.action_dim)  # should be overridden

        self.nu = np.ones(self.action_dim) * np.log(init_std)

    def sample_action(self):
        std = np.sqrt(np.exp(self.nu))
        action_flat = self.mean + std * np.random.randn(self.action_dim)
        return action_flat.reshape((2, self.Ng))

    def initialize_baseline(self, env, n_trials=100):
        rewards = []
        actions = []
        for _ in range(n_trials):
            env.reset()
            a  = self.sample_action()
            _, r, _, _ = env.step(a)
            rewards.append(r)
            actions.append(a)

        self.rwd_baseline = np.mean(rewards)
        return np.mean(rewards)

    def update(self, action, reward):
        """
        Policy gradient update

        Parameters:
            actions
            rewards: list of scalar rewards
        """
        # update reward baseline
        self.rwd_baseline = self.baseline_decay * self.rwd_baseline + (1 - self.baseline_decay) * reward
        
        # flatten action matrix into a vector
        action_flat = action.flatten()
        delta = action_flat - self.mean
        cov = np.diag(np.exp(self.nu)) # covariance matrix
        
        grad_logp_mu = np.linalg.inv(cov) @ delta.T
        self.mean += (self.alpha * (reward - self.rwd_baseline) * grad_logp_mu).ravel()
        
        grad_logp_nu = -.5 +.5*(delta **2) * np.exp(-self.nu) 
        self.nu += self.alpha_nu * grad_logp_nu.ravel() * (reward - self.rwd_baseline)
        
    def get_policy_mean(self):
        return self.mean.reshape((2, self.Ng))
