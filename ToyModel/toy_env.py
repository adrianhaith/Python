import numpy as np


class Toy2DEnv:
    def __init__(self):
        self.u1_range = (0.0, 1.0)
        self.u2_range = (0.0, 0.7)
        self.action_space = np.array([self.u1_range, self.u2_range])  # for convenience
        self.observation_space = None  # no state

    def reward(self, u1, u2):
        # return np.exp(-20 * (u2 - (0.7 - u1)**2))
        # return 2*np.exp(-10*(u2 - (0.9 - (u1 - 1)**2))**2)-.95
        y_valley = -1*(u1-0.3)*(u1-1.4)
        return np.exp(-2*(u1-.7)**2)*np.exp(-10*(u2-y_valley)**2)
    

    def reset(self):
        """Resets the environment. Returns a dummy observation."""
        return None

    def step(self, action):
        """Takes a 2D action and returns reward."""
        u1, u2 = action
        r = self.reward(u1, u2)
        obs = None  # stateless
        done = True  # single-step environment
        info = {}
        return obs, r, done, info

    def get_reward_grid(self, resolution=100):
        """Return meshgrid and reward grid for visualization."""
        u1 = np.linspace(*self.u1_range, resolution)
        u2 = np.linspace(*self.u2_range, resolution)
        U1, U2 = np.meshgrid(u1, u2)
        R = self.reward(U1, U2)
        return U1, U2, R
