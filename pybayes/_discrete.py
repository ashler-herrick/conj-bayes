from scipy import stats
import numpy as np



#class for binomial likelihood  
class binomial:
    #prior must be Beta for conjugacy so we exclude that parameter here
    def __init__(self, alpha_0 = 2, beta_0 = 2, m = 10):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        #use m as number of trials instead of n, which is reserved for number of datapoints
        self.m = m

    def update_model(self, data, alpha_0 = None, beta_0 = None, m = None):
        if alpha_0 != None:
            self.alpha_0 = alpha_0
        if beta_0 != None:
            self.beta_0 = beta_0
        if m != None:
            self.m = m

        n = len(data)

        self.alpha_n = self.alpha_0 + np.sum(data)
        self.beta_n = self.beta_0 + n * self.m - np.sum(data)

    def sample_posterior_predictive(self, n = 1, seed = None):
        return stats.betabinom.rvs(n = self.m, a = self.alpha_n, b = self.beta_n, size = n, random_state = seed)
    
#bernoulli likelihood
class bernoulli:
    #prior is always beta
    def __init__(self,alpha_0 = 2, beta_0 = 2):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

    def update_model(self, data, alpha_0 = None, beta_0 = None):
        if alpha_0 != None:
            self.alpha_0 = alpha_0
        if beta_0 != None:
            self.beta_0 = beta_0

        n = len(data)

        self.alpha_n = self.alpha_0 + np.sum(data)
        self.beta_n = self.beta_0 + n - np.sum(data)

        self.p_n = self.alpha_n/(self.alpha_n + self.beta_n)

    def sample_posterior_predictive(self, n = 1, seed = None):
        return stats.bernoulli.rvs(p = self.p_n, size = n, random_state = seed)
