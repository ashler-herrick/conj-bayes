from scipy import stats
import numpy as np
from ._model_infra import model


#class for binomial likelihood  
class binomial(model):

    def update_model(self, data, **params):
        super()._update_model(**params)
        self._check_params(['alpha_0','beta_0','m'])

        n = len(data)

        self.alpha_n = self.alpha_0 + np.sum(data)
        self.beta_n = self.beta_0 + n * self.m - np.sum(data)
    
    def posterior_mode(self):
        return (self.alpha_n - 1)/(self.alpha_n + self.beta_n - 2)
    
    def posterior_mean(self):
        return self.alpha_n/(self.alpha_n + self.beta_n)

    def sample_posterior(self, n = 1, seed = None):
        return stats.beta.rvs(a = self.alpha_n, b = self.alpha_n, size = n, random_state = seed)

    def sample_prior_predictive(self, n = 1, seed = None):
        self._check_params(['alpha_0','beta_0','m'])
        return stats.betabinom.rvs(n = self.m, a = self.alpha_0, b = self.beta_0, size = n, random_state = seed)
    
    def sample_posterior_predictive(self, n = 1, seed = None):
        return stats.betabinom.rvs(n = self.m, a = self.alpha_n, b = self.beta_n, size = n, random_state = seed)
    
#bernoulli likelihood
class bernoulli(model):

    def update_model(self, data, **params):
        super()._update_model(**params)
        self._check_params(['alpha_0','beta_0'])

        self.p_0 = self.alpha_0/ (self.alpha_0 + self.beta_0)
        n = len(data)

        self.alpha_n = self.alpha_0 + np.sum(data)
        self.beta_n = self.beta_0 + n - np.sum(data)

        self.p_n = self.alpha_n/(self.alpha_n + self.beta_n)

    def posterior_mode(self):
        return (self.alpha_n - 1)/(self.alpha_n + self.beta_n - 2)
    
    def posterior_mean(self):
        return self.alpha_n/(self.alpha_n + self.beta_n)
    
    def sample_posterior(self, n = 1, seed = None):
        return stats.beta.rvs(a = self.alpha_n, b = self.alpha_n, size = n, random_state = seed)
    
    def sample_prior_predictive(self, n = 1, seed = None):
        self._check_params(['alpha_0','beta_0'])
        return stats.bernoulli.rvs(p = self.p_n, size = n, random_state = seed)

    def sample_posterior_predictive(self, n = 1, seed = None):
        return stats.bernoulli.rvs(p = self.p_n, size = n, random_state = seed)
    
#negative binomial likelihood
class negative_binomial:
    
    def update_model(self, data, **params):
        super()._update_model(**params)
        self._check_params(['alpha_0','beta_0','r'])

        n = len(data)

        self.alpha_n = self.alpha_0 + self.r * n
        self.beta_n = self.beta_0 + np.sum(data)

    def posterior_mode(self):
        return (self.alpha_n - 1)/(self.alpha_n + self.beta_n - 2)
    
    def posterior_mean(self):
        return self.alpha_n/(self.alpha_n + self.beta_n)
    
    def sample_posterior(self, n = 1, seed = None):
        return stats.beta.rvs(a = self.alpha_n, b = self.alpha_n, size = n, random_state = seed)
    
    def sample_prior_predictive(self, n = 1, seed = None):
        self._check_params(['alpha_0','beta_0'])
        random_state = np.random.RandomState(seed)
        p = random_state.beta(a = self.alpha_0, b = self.beta_0, size = n)
        return random_state.negative_binomial(n = self.r, p = p, size = n)

    def sample_posterior_predictive(self, n = 1, seed = None):
        random_state = np.random.RandomState(seed)
        p = random_state.beta(a = self.alpha_n, b = self.beta_n, size = n)
        return random_state.negative_binomial(n = self.r, p = p, size = n)


    
