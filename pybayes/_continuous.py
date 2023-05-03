from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from _model_infra import model

#class for multivariate likelihood
class multivariate_normal(model):

    def update_model(self, data, **params):
        super()._update_model(**params)
        if not hasattr(self,'prior'):
                print("Please enter a valid prior distribution: 'norm_inv_wishart' or 'multiviariate_normal'")
        if self.prior == 'norm_inv_wishart':
            self._check_params(['kappa_0','nu_0','mu_0','Sigma_0'])
        if self.prior == 'multivariate_normal':
            self._check_params(['mu_0','Sigma_0','Sigma'])
        
        #make sure data is a numpy array
        data = np.asarray(data)

        #get n and d
        n = len(data)
        self.d = len(data[0])
        
        #assumes each observation is a row
        x_bar = np.mean(data,axis = 0)

        if self.prior == 'multivariate_normal':


            #Covariance matrix related calculations
            self.Sigma = Sigma
            Sigma_inv = np.linalg.inv(Sigma)
            Sigma_0_inv = np.linalg.inv(Sigma_0)

            #Sigma_n is the covariance matrix of the distribution of mean
            self.Sigma_n = np.linalg.inv(Sigma_0_inv + n * Sigma_inv)

            #mu_n is the mean
            self.mu_n = self.Sigma_n @ (n * Sigma_inv @ x_bar + Sigma_0_inv @ mu_0)

            #create var for the posterior mode of the mean vector
            self.post_mode_mu = self.mu_n

            #calculate posterior predictive parameters
            self.post_pred_mean = self.mu_n
            self.post_pred_cov = self.Sigma + self.Sigma_n 

        if self.prior == 'norm_inv_wishart':


            #these are used as weights
            self.kappa_n = self.kappa_0 + n
            self.nu_n = self.nu_0 + n

            #our initial value for the sum of squares matrix is the initial value of Sigma_0 times the factor that is divided in the post pred
            S_0 = (self.nu_0 - self.d + 1)*Sigma_0

            #weighted average of of prior and sample    
            self.mu_n = (1/(self.kappa_n)) * (self.kappa_0 * self.mu_0 + n * x_bar)

            #calculate sum of squares matrix
            res = data - x_bar
            if len(res) == 1:
                S = np.zeros((self.d,self.d))
            else:
                S = np.sum([np.outer(x,x) for x in res], axis = 0)

            #first term is init value, second term is sample sum of squares, third term is penalty
            self.S_n = S_0 + S + np.outer(x_bar - mu_0,x_bar - mu_0)*((self.kappa_0 * n)/(self.kappa_n))

            #store posterior stats as attributes
            self.post_mode_mu = self.mu_n
            self.post_mode_Sigma = self.S_n * (self.nu_n + self.d + 1)
            self.post_pred_mean = self.mu_n
            self.post_pred_cov = self.S_n * (self.kappa_n + 1)/(self.kappa_n * (self.nu_n - self.d + 1))
            self.post_pred_df = self.nu_n - self.d + 1

    def sample_posterior_predictive(self, n = 1, seed = None):
        if self.prior == 'multivariate_normal':
            return stats.multivariate_normal(mean = self.mu_n, cov = self.post_pred_cov, size = n, random_state = seed)     
        if self.prior == 'norm_inv_wishart':
            return stats.multivariate_t(loc = self.post_pred_mean, scale = self.post_pred_cov, df = self.post_pred_df, size = n, random_state = seed)

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

    def update_model(self, data, kappa_0 = None, nu_0 = None, mu_0 = None, sigma_sq_0 = None, sigma_sq = None):
        #if values are passed then update the values from initialization

        #TODO: find a way to automate all this so I don't have to write so many lines
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

