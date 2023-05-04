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
            Sigma_inv = np.linalg.inv(self.Sigma)
            Sigma_0_inv = np.linalg.inv(self.Sigma_0)

            #Sigma_n is the covariance matrix of the distribution of mean
            self.Sigma_n = np.linalg.inv(Sigma_0_inv + n * Sigma_inv)

            #mu_n is the mean
            self.mu_n = self.Sigma_n @ (n * Sigma_inv @ x_bar + Sigma_0_inv @ self.mu_0)

        if self.prior == 'norm_inv_wishart':


            #these are used as weights
            self.kappa_n = self.kappa_0 + n
            self.nu_n = self.nu_0 + n

            #our initial value for the sum of squares matrix is the initial value of Sigma_0 times the factor that is divided in the post pred
            S_0 = (self.nu_0 - self.d + 1)*self.Sigma_0

            #weighted average of of prior and sample    
            self.mu_n = (1/(self.kappa_n)) * (self.kappa_0 * self.mu_0 + n * x_bar)

            #calculate sum of squares matrix
            res = data - x_bar
            if len(res) == 1:
                S = np.zeros((self.d,self.d))
                print('Warning: Unable to calculate scatter matrix with one data point, posterior covariance matrix will equal prior.')
            else:
                S = np.sum([np.outer(x,x) for x in res], axis = 0)

            #first term is init value, second term is sample sum of squares, third term is penalty
            self.S_n = S_0 + S + np.outer(x_bar - self.mu_0,x_bar - self.mu_0)*((self.kappa_0 * n)/(self.kappa_n))

    
    def posterior_mode(self):
        if self.prior == 'norm_inv_wishart':
            return {'mu_n' : self.mu_n, 'Sigma_n': self.S_n / (self.nu_n + self.d + 1)}
        if self.prior == 'multivariate_normal':
            return self.mu_n
    
    def posterior_mean(self):
        if self.prior == 'norm_inv_wishart':
            return {'mu_n' : self.mu_n, 'Sigma_n': self.S_n / (self.nu_n - self.d - 1)}
        if self.prior == 'multivariate_normal':
            return self.mu_n

    def sample_posterior(self, param = None, n = 1, seed = None):
        if self.prior == 'norm_inv_wishart':
            mu_rvs = stats.multivariate_t.rvs(loc = self.mu_n, shape = self.S_n/(self.kappa_n * (self.nu_n - self.d + 1)), df = self.nu_n - self.d + 1, random_state = seed, size = n)
            Sigma_rvs = stats.invwishart.rvs(df = self.nu_n, scale = self.S_n, random_state = seed, size = n)
            return {'mu' : mu_rvs, 'Sigma': Sigma_rvs}

        if self.prior == 'mutlivariate_normal':
            return stats.multivariate_normal.rvs(mean = self.mu_n, cov = self.Sigma_n, size = n, random_state = seed)
            
        
    def sample_prior(self, n = 1, seed = None):

        if self.prior == 'norm_inv_wishart':
            self._check_params(['mu_0','S_0','d','kappa_0','nu_0'])
            mu_rvs = stats.multivariate_t.rvs(loc = self.mu_0, shape = self.S_0/(self.kappa_0 * (self.nu_0 - self.d + 1)), df = self.nu_0 - self.d + 1, random_state = seed, size = n)
            Sigma_rvs = stats.invwishart.rvs(df = self.nu_0, scale = self.S_0, random_state = seed, size = n)
            return {'mu' : mu_rvs, 'Sigma': Sigma_rvs}

        if self.prior == 'mutlivariate_normal':
            self._check_params(['mu_0','Sigma_0'])
            return stats.multivariate_normal.rvs(mean = self.mu_0, cov = self.Sigma, size = n, random_state = seed)

    def sample_prior_predictive(self, n = 1, seed = None):
        if self.prior == 'multivariate_normal':
            self._check_params(['mu_0','Sigma'])
            return stats.multivariate_normal.rvs(mean = self.mu_0, cov = self.Sigma, size = n, random_state = seed)     
        
        if self.prior == 'norm_inv_wishart':
            self._check_params(['mu_0','S_0','d','kappa_0','nu_0'])
            return stats.multivariate_t.rvs(loc = self.mu_0, scale = self.S_0 * (self.kappa_0 + 1)/(self.kappa_0 * (self.nu_0 - self.d + 1)), df = self.d, size = n, random_state = seed)
    
    def sample_posterior_predictive(self, n = 1, seed = None):
        if self.prior == 'multivariate_normal':
            return stats.multivariate_normal.rvs(mean = self.mu_n, cov = self.Sigma, size = n, random_state = seed)     
        
        if self.prior == 'norm_inv_wishart':
            return stats.multivariate_t.rvs(loc = self.mu_n, scale = self.S_n * (self.kappa_n + 1)/(self.kappa_n * (self.nu_n - self.d + 1)), df = self.d, size = n, random_state = seed)


class normal:

    def update_model(self, data, **params):
        super()._update_model(**params)
        if not hasattr(self,'prior'):
                print("Please enter a valid prior distribution: 'norm_inv_chi_sq' or 'normal'")
        if self.prior == 'norm_inv_chi_sq':
            self._check_params(['kappa_0','nu_0','mu_0','sigma_0'])
        if self.prior == 'normal':
            self._check_params(['mu_0','sigma_0','sigma'])

        n = len(data)
        x_bar = np.mean(data)

        if self.prior == 'normal':

            self.sigma_n = np.sqrt(1/(n/self.sigma**2 + 1/self.sigma_0**2))

            self.mu_n = self.sigma_n**2 * (self.mu_0/self.sigma_0**2 + (n*x_bar)/self.sigma**2)

        if self.prior == 'norm_inv_chi_sq':
            #update the weights
            self.kappa_n = self.kappa_0 + n
            self.nu_n = self.nu_0 + n

            #mu_n is a weighted average of prior and sample means
            self.mu_n = (self.kappa_0 *  self.mu_0 + n * x_bar)/self.kappa

            res = data - x_bar
            ss = np.sum(res**2)
            #posterior variance is prior sum of squares plus sample sum of squares plus a penalty term
            self.sigma_n = np.sqrt((self.nu_0 * self.sigma_0**2 + ss + ((n*self.kappa_0)/(self.kappa_n))*(self.mu_0 - x_bar)**2)/self.nu_n)


   
    def posterior_mode(self):
        if self.prior == 'norm_inv_chi_sq':
            return {'mu_n' : self.mu_n, 'sigma_n^2': (self.nu_n * self.sigma_n**2)/(self.nu_n - 1)}
        if self.prior == 'normal':
            return self.mu_n
    
    def posterior_mean(self):
        if self.prior == 'norm_inv_chi_sq':
            return {'mu_n' : self.mu_n, 'sigma_n^2': (self.nu_n * self.sigma_n**2)/(self.nu_n - 2)}
        if self.prior == 'normal':
            return self.mu_n
        
    def sample_posterior(self, n = 1, seed = None):
        if self.prior == 'norm_inv_chi_sq':
            mu_rvs = stats.t.rvs(loc = self.mu_n, scale = self.sigma_n**2 /self.kappa_n, df = self.nu_n, random_state = seed, size = n)
            sigma_rvs = stats.chi2.rvs(df = self.nu_n, scale = self.sigma_n**2, random_state = seed, size = n)
            return {'mu' : mu_rvs, 'sigma': sigma_rvs}
        
        if self.prior == 'normal':
            return stats.norm.rvs(loc = self.mu_n, scale = self.sigma**2, size = n, random_state = seed)
            
        
    def sample_prior(self, n = 1, seed = None):

        if self.prior == 'norm_inv_chi_sq':
            self._check_params(['kappa_0','nu_0','mu_0','sigma_0'])
            mu_rvs = stats.t.rvs(loc = self.mu_n, scale = self.sigma_n**2 /self.kappa_n, df = self.nu_n, random_state = seed, size = n)
            sigma_rvs = stats.chi2.rvs(df = self.nu_n, scale = self.sigma_n**2, random_state = seed, size = n)
            return {'mu' : mu_rvs, 'sigma': sigma_rvs}
        
        if self.prior == 'normal':
            self._check_params(['mu_0','sigma_0','sigma'])
            return stats.norm.rvs(loc = self.mu_n, scale = self.sigma**2, size = n, random_state = seed)

    def sample_prior_predictive(self, n = 1, seed = None):
        if self.prior == 'normal':
            self._check_params(['mu_0','sigma'])
            return stats.norm.rvs(loc = self.mu_0, scale = self.sigma, size = n, random_state = seed)     
        
        if self.prior == 'norm_inv_chi_sq':
            self._check_params(['mu_0','sigma_0','d','kappa_0','nu_0'])
            return stats.t.rvs(loc = self.mu_0, scale = ((1 + self.kappa_0)/self.kappa_0) * self.sigma_0**2, df = self.d, size = n, random_state = seed)
    
    def sample_posterior_predictive(self, n = 1, seed = None):
        if self.prior == 'normal':
            return stats.norm.rvs(loc = self.mu_n, scale = self.sigma, size = n, random_state = seed)     
        
        if self.prior == 'norm_inv_chi_sq':
            self._check_params(['mu_n','sigma_n','d','kappa_n','nu_n'])
            return stats.t.rvs(loc = self.mu_n, scale = ((1 + self.kappa_n)/self.kappa_n) * self.sigma_n**2, df = self.d, size = n, random_state = seed)

    #helper function for the plotting with the norm inverse chi squared prior
    def _plot_norm_inv_chi_sq(self, mu, sigma, kappa, nu, label):
        #give some extra width since its student t
        x = np.linspace(-4 * sigma, 4 * sigma, 100)
        plt.plot(stats.t.pdf(loc = mu, scale = sigma**2/kappa, df = nu), label = label)
        self._showplot(xlab = 'mu', ylab = 'pdf',)
        #scaled inv chi squared(nu,sigma^2) is inv-gamma with alpha = nu/2 and beta = (nu * sigma^2)/2
        #our space will cover from 0 to 3x the mean
        x = np.linspace(0,3 * (nu * sigma**2)/(nu - 2))
        plt.plot(stats.invgamma.pdf(x = x, a = nu/2, b = (nu * sigma**2)/2), label = label)
        self._showplot(xlab = 'sigma^2', ylab = 'pdf')

    def plot(self, plot_type = None):
        

        if self.prior == 'normal':

            self._check_plot(['mu_0','sigma_0','mu_n','sigma_n'], plot_type)
            x_prior = np.linspace(-3 * self.sigma_0, 3 * self.sigma_0, 100)
            x_post = np.linspace(-3 * self.sigma_n, 3 * self.sigma_n, 100)

            if self.plot_type == 'prior':
                self._check_plot(['mu_0','sigma_0'], plot_type)
                plt.plot(x_prior,stats.norm.pdf(x_prior, loc = self.mu_0, scale = self.sigma_0), label = 'Prior')
        
            if plot_type == 'posterior':
                self._check_plot(['mu_n','sigma_n'], plot_type)
                plt.plot(x_post,stats.norm.pdf(x_post, loc = self.mu_n, scale = self.sigma_n), label = 'Posterior')

            if plot_type == 'both':
                self._check_plot(['mu_0','sigma_0','mu_n','sigma_n'], plot_type)
                plt.plot(x_prior,stats.norm.pdf(x_prior, loc = self.mu_0, scale = self.sigma_0), label = 'Prior')
                plt.plot(x_post,stats.norm.pdf(x_post, loc = self.mu_n, scale = self.sigma_n), label = 'Posterior')
        self._showplot(xlab = 'mu', ylab = 'pdf')

        #Maybe add multivariate plotting here? 
        if self.prior == 'norm_inv_chi_sq':
            self._check_plot(['mu_0','sigma_0','kappa_0','nu_0','mu_n','sigma_n','kappa_n','nu_n'])
            x_post = np.linspace(-4 * self.sigma_n, 4 * self.sigma_n, 100)
            if plot_type == 'prior':
                self._check_plot(['mu_0','sigma_0','kappa_0','nu_0'])
                self._plot_norm_inv_chi_sq(mu = self.mu_0, sigma = self.sigma_0, kappa = self.kappa_0, nu = self.nu_0, label = 'Prior')
                
            if plot_type == 'posterior':
                self._check_plot(['mu_n','sigma_n','kappa_n','nu_n'])
                self._plot_norm_inv_chi_sq(mu = self.mu_n, sigma = self.sigma_n, kappa = self.kappa_n, nu = self.nu_n, label = 'Posterior')

            if plot_type == 'both':
                self._check_plot(['mu_0','sigma_0','kappa_0','nu_0','mu_n','sigma_n','kappa_n','nu_n'])







    

