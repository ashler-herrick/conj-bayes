import numpy as np
from scipy import stats


class normal:
    #default to standard normal
    def __init__(self, mu_0 = 0, sigma_sq_0 = 1, sigma_sq = 1):
        self.mu_0 = mu_0
        self.sigma_sq = 1
        self.sigma_sq_0 = sigma_sq_0
        #also define the precision
        self.tau_0 = 1/sigma_sq_0
        self.tau = 1/sigma_sq
    def update_params(self, data, mu_0 = None, sigma_sq_0 = None, sigma_sq = None):
        if mu_0 == None:
            mu_0 = self.mu_0
        if sigma_sq == None:
            sigma_sq = self.sigma_sq
        if sigma_sq_0 == None:
            sigma_sq_0 = self.sigma_sq_0


        n = len(data)
        x_bar = np.mean(data)
        self.tau_n = self.tau_0 + n * self.tau
        self.sigma_sq_n = 1/self.tau_n
        self.mu_n = (x_bar * n * self.tau + self.mu_0 * self.tau_0)/self.tau_n

        self.post_mode_mu = self.mu_n

        self.post_pred_mean = self.mu_n
        self.post_pred_var = self.sigma_sq_n + self.sigma_sq

    def sample_posterior_predictive(self,n = 1, seed = None):
        return stats.norm.rvs(loc = self.post_pred_mean, scale = self.post_pred_var, size = n, random_state = seed)