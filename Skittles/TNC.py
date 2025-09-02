# define a class to implement the TNC-cost analysis from Muller & Sternad, 2009, EBR
import numpy as np

class TNCCost:
    def __init__(self, env, n_grid=60, n_collapse=100, n_samples=60):
        """
        env: Skittles environment with method `compute_error(action)`
        n_grid: resolution of grid search for T-Cost
        n_collapse: steps for collapsing dataset toward mean for N-Cost
        n_samples: trials per block (default = 60, like in paper)
        """
        self.env = env
        self.n_grid = n_grid
        self.n_collapse = n_collapse
        self.n_samples = n_samples

    def mean_error(self, actions):
        errors = []
        for action in actions:
            error, _ = self.env.compute_error(action)  # keep only the error
            errors.append(error)
        return np.mean(errors)

    # -------------------------------
    # T-Cost: shift distribution to most tolerant region
    # -------------------------------
    def t_cost(self, actions):
        mu = np.mean(actions, axis=0)
        cov = np.cov(actions, rowvar=False)
        actual_err = self.mean_error(actions)

        # Define grid limits based on task (adjust to your env)
        angle_range = np.linspace(0, -180*np.pi/180, self.n_grid)
        vel_range   = np.linspace(200*np.pi/180, 800*np.pi/180, self.n_grid)

        best_err = np.inf
        for a in angle_range:
            for v in vel_range:
                shifted_actions = np.random.multivariate_normal([a, v], cov, size=self.n_samples)
                err = self.mean_error(shifted_actions)
                if err < best_err:
                    best_err = err

        return actual_err - best_err

    # -------------------------------
    # N-Cost: collapse variability radially toward mean
    # -------------------------------
    def n_cost(self, actions):
        mu = np.mean(actions, axis=0)
        actual_err = self.mean_error(actions)

        best_err = np.inf
        for k in range(1, self.n_collapse+1):
            alpha = 1 - k/self.n_collapse
            shrunk_actions = mu + alpha * (actions - mu)
            err = self.mean_error(shrunk_actions)
            if err < best_err:
                best_err = err

        return actual_err - best_err

    # -------------------------------
    # C-Cost: optimize covariation via greedy pair-swapping
    # -------------------------------
    def c_cost(self, actions):
        actual_err = self.mean_error(actions)
        optimized = actions.copy()
        improved = True

        while improved:
            improved = False
            for i in range(len(optimized)):
                for j in range(i+1, len(optimized)):
                    swapped = optimized.copy()
                    swapped[i,1], swapped[j,1] = optimized[j,1], optimized[i,1]  # swap velocities

                    if self.mean_error(swapped) < self.mean_error(optimized):
                        optimized = swapped
                        improved = True

        best_err = self.mean_error(optimized)
        return actual_err - best_err

    # -------------------------------
    # Wrapper: compute all three costs
    # -------------------------------
    def compute_all(self, actions):
        return {
            "T-Cost": self.t_cost(actions),
            "N-Cost": self.n_cost(actions),
            "C-Cost": self.c_cost(actions),
            "MeanError": self.mean_error(actions)
        }
