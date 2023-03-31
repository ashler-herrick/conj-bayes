from scipy import stats
import numpy as np

#class for normal likelihood
class normal:
    #default to standard normal with reference prior
    def __init__(self, prior, kappa_0 = 0, nu_0 = -1,mu_0 = 0, sigma_sq_0 = 1, sigma_sq = 1):
        if prior not in ('normal','norm_inv_chi_sq'):
            print("Warning: invalid prior entered. Choose either 'normal' or 'norm_inv_chi_sq'")
        self.prior = prior
        self.mu_0 = mu_0
        self.sigma_sq = 1
        self.sigma_sq_0 = sigma_sq_0
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0
        #also define the precision
        self.tau_0 = 1/sigma_sq_0
        self.tau = 1/sigma_sq

    def update_params(self, data, kappa_0 = None, nu_0 = None, mu_0 = None, sigma_sq_0 = None, sigma_sq = None):
        #if values are passed then update the values from initialization
        if mu_0 != None:
            self.mu_0 = mu_0 
        if sigma_sq != None:
            self.sigma_sq = sigma_sq
        if sigma_sq_0 != None:
            self.sigma_sq_0 = sigma_sq_0
        if kappa_0 != None:
            self.kappa_0 = kappa_0
        if nu_0 != None:
            self.nu_0 = nu_0
        #also update the precision
        self.tau_0 = 1/self.sigma_sq_0
        self.tau = 1/self.sigma_sq

        n = len(data)
        x_bar = np.mean(data)

        if self.prior == 'normal':

            self.tau_n = self.tau_0 + n * self.tau
            self.sigma_sq_n = 1/self.tau_n
            self.mu_n = (x_bar * n * self.tau + self.mu_0 * self.tau_0)/self.tau_n

            #store posterior predictive parameters
            self.post_mode_mean = self.mu_n
            self.post_pred_var = self.sigma_sq_n + self.sigma_sq

        if self.prior == 'norm_inv_chi_sq':
            #update the weights
            self.kappa_n = self.kappa_0 + n
            self.nu_n = self.nu_0 + n

            #mu_n is a weighted average of prior and sample means
            self.mu_n = (self.kappa_0 *  self.mu_0 + n * x_bar)/self.kappa

            ss = np.sum([(x_i - x_bar)**2 for x_i in data])
            #posterior variance is prior sum of squares plus sample sum of squares plus a penalty term
            self.sigma_sq_n = (self.nu_0 * self.sigma_sq_0 + ss + ((n*self.kappa_0)/(self.kappa_n))*(self.mu_0 - x_bar)**2)/self.nu_n

            self.post_pred_mean = self.mu_n
            self.post_pred_scale = self.sigma_sq_n * (self.kappa_n + 1)/self.kappa_n
            self.post_pred_df = self.nu_n
            


    def sample_posterior_predictive(self,n = 1, seed = None):
        if self.prior == 'normal':
            return stats.norm.rvs(loc = self.post_pred_mean, scale = self.post_pred_var, size = n, random_state = seed)
        if self.prior == 'norm_inv_chi_sq':
            return stats.t.rvs(loc = self.post_pred_mean, scale = self.post_pred_scale, df = self.post_pred_df, size = n, random_state = seed)