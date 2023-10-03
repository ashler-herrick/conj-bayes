from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from conj_bayes._model_infra import model


#==================================================
#binomial likelihood
#==================================================
class binomial(model):

    def update_model(self, data, **params):

        super()._update_model(**params)
        self._check_params(['alpha_0','beta_0','m'])

        data = np.asarray(data)
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
    
    def plot(self, plot_type):
        x = np.linspace(0,1,100)
        if plot_type == 'prior':
            self._check_plot(['alpha_0','beta_0','alpha_n','beta_n'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_0, b = self.beta_0), label = 'Prior')
        
        if plot_type == 'posterior':
            self._check_plot(['alpha_n','beta_n'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_n, b = self.beta_n), label = 'Posterior')

        if plot_type == 'both':
            self._check_plot(['alpha_0','beta_0','alpha_n','beta_n'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_0, b = self.beta_0), label = 'Prior')
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_n, b = self.beta_n), label = 'Posterior')

        plt.xlabel('p')
        plt.ylabel('pdf')
        plt.legend()
        plt.title("Distribution of p")
        plt.show()



    
#==================================================
#bernoulli likelihood
#==================================================
class bernoulli(model):

    def update_model(self, data, **params):
        super()._update_model(**params)
        self._check_params(['alpha_0','beta_0'])

        
        data = np.asarray(data)
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
        self.p_0 = self.alpha_0/ (self.alpha_0 + self.beta_0)
        return stats.bernoulli.rvs(p = self.p_0, size = n, random_state = seed)

    def sample_posterior_predictive(self, n = 1, seed = None):
        return stats.bernoulli.rvs(p = self.p_n, size = n, random_state = seed)
    
    def plot(self, plot_type):
        x = np.linspace(0,1,100)
        if plot_type == 'prior':
            self._check_plot(['alpha_0','beta_0'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_0, b = self.beta_0), label = 'Prior')
        
        if plot_type == 'posterior':
            self._check_plot(['alpha_n','beta_n'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_n, b = self.beta_n), label = 'Posterior')

        if plot_type == 'both':
            self._check_plot(['alpha_0','beta_0','alpha_n','beta_n'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_0, b = self.beta_0), label = 'Prior')
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_n, b = self.beta_n), label = 'Posterior')


        plt.xlabel('p')
        plt.ylabel('pdf')
        plt.legend()
        plt.title("Distribution of p")
        plt.show()
    
#==================================================
#negative_binomial likelihood
#==================================================
class negative_binomial(model):
    
    def update_model(self, data, **params):
        super()._update_model(**params)
        self._check_params(['alpha_0','beta_0','r'])

        data = np.asarray(data)
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
    
    def plot(self, plot_type):
        x = np.linspace(0,1,100)
        if plot_type == 'prior':
            self._check_plot(['alpha_0','beta_0','alpha_n','beta_n'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_0, b = self.beta_0), label = 'Prior')
        
        if plot_type == 'posterior':
            self._check_plot(['alpha_n','beta_n'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_n, b = self.beta_n), label = 'Posterior')

        if plot_type == 'both':
            self._check_plot(['alpha_0','beta_0','alpha_n','beta_n'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_0, b = self.beta_0), label = 'Prior')
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_n, b = self.beta_n), label = 'Posterior')


        plt.xlabel('p')
        plt.ylabel('pdf')
        plt.legend()
        plt.title("Distribution of p")
        plt.show()


#==================================================
#poisson likelihood
#==================================================
class poisson(model):

    def update_model(self, data, **params):
        super()._update_model(**params)
        self._check_params(['alpha_0','beta_0'])

        data = np.asarray(data)
        n = len(data)

        self.alpha_n = self.alpha_0 + np.sum(data)
        self.beta_n = self.beta_0 + n

    def posterior_mode(self):
        return (self.alpha_n - 1)/self.beta_n
    
    def posterior_mean(self):
        return self.alpha_n/self.beta_n
    
    def sample_prior(self, n = 1, seed = None):
        random_state = np.random.RandomState(seed)
        return random_state.gamma(shape = self.alpha_0, scale = 1/self.beta_0, size = n)
     
    def sample_posterior(self, n = 1, seed = None):
        random_state = np.random.RandomState(seed)
        return random_state.gamma(shape = self.alpha_n, scale = 1/self.beta_n, size = n)
    
    def sample_prior_predictive(self, n = 1, seed = None):
        random_state = np.random.RandomState(seed)
        return  random_state.negative_binomial(n = self.alpha_0, p = self.beta_0/(self.beta_0 + 1), size = n)
    
    def sample_posterior_predictive(self, n = 1, seed = None):
        random_state = np.random.RandomState(seed)
        return  random_state.negative_binomial(n = self.alpha_n, p = self.beta_n/(self.beta_n + 1), size = n)
    
    def plot(self, plot_type):
        self._check_plot(['alpha_0','beta_0','alpha_n','beta_n'], plot_type)

        x = np.linspace(0,2 * self.alpha_n/self.beta_n,100)
        if plot_type == 'prior':
            plt.plot(x,stats.gamma.pdf(x, a = self.alpha_0, scale = 1/self.beta_0), label = 'Prior')
        
        if plot_type == 'posterior':
            plt.plot(x,stats.gamma.pdf(x, a = self.alpha_n, scale = 1/self.beta_n), label = 'Posterior')

        if plot_type == 'both':
            plt.plot(x,stats.gamma.pdf(x, a = self.alpha_0, scale = 1/self.beta_0), label = 'Prior')
            plt.plot(x,stats.gamma.pdf(x, a = self.alpha_n, scale = 1/self.beta_n), label = 'Posterior')


        plt.xlabel('lambda')
        plt.ylabel('pdf')
        plt.legend()
        plt.title("Distribution of lambda")
        plt.show()

#==================================================
#categorial likelihood
#==================================================
class categorical(model):

    def update_model(self, data, **params):
        super()._update_model(**params)
        self._check_params(['alpha_0'])

        data = np.asarray(data)

        self.alpha_n = self.alpha_0 + np.sum(data,axis = 0)

    def posterior_mean(self):
        return self.alpha_n/np.sum(self.alpha_n)
     
    def posterior_mode(self):
        return (self.alpha_n - 1)/(np.sum(self.alpha_n)-len(self.alpha_n))
    
    def sample_prior_predictive(self,n = 1, seed = None):
        self._check_params(['alpha_0'])

        self.p_0 = self.alpha_0/np.sum(self.alpha_0)
        random_state = np.random.RandomState(seed)
        return random_state.choice(np.ones(len(self.p_0)), p = self.p_0)
    
    def sample_posterior_predictive(self, n = 1, seed = None):
        self.p = self.alpha_n/np.sum(self.alpha_n)
        random_state = np.random.RandomState(seed)
        return random_state.choice(np.ones(len(self.p)), p = self.p)

#==================================================
#multinomial likelihood
#==================================================    
class multinomial(model):

    def update_model(self, data, **params):
        super()._update_model(**params)
        self._check_params(['alpha_0','n'])

        data = np.asarray(data)

        self.alpha_n = self.alpha_0 + np.sum(data,axis = 0)

    def posterior_mean(self):
        return self.alpha_n/np.sum(self.alpha_n)
     
    def posterior_mode(self):
        return (self.alpha_n - 1)/(np.sum(self.alpha_n)-len(self.alpha_n))
    
    def sample_prior_predictive(self,n = 1, seed = None):
        self._check_params(['alpha_0','n'])

        random_state = np.random.RandomState(seed)
        p = random_state.dirichlet(self.alpha_0)
        return random_state.choice(np.ones(len(self.alpha_0)), p = p)
    
    def sample_posterior_predictive(self, n = 1, seed = None):
        random_state = np.random.RandomState(seed)
        p = random_state.dirichlet(self.alpha_n)
        return random_state.choice(np.ones(len(self.alpha_n)), p = p)

#==================================================
#geometric likelihood
#==================================================    
class geometric(model):

    def update_model(self, data, **params):
        super()._update_model(**params)
        self._check_params(['alpha_0','beta_0'])

        data = np.asarray(data)
        n = len(data)
        self.alpha_n = self.alpha_0 + n
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
        return random_state.geometric(p = p, size = n)

    def sample_posterior_predictive(self, n = 1, seed = None):
        self._check_params(['alpha_0','beta_0','alpha_n','beta_n'])
        random_state = np.random.RandomState(seed)
        p = random_state.beta(a = self.alpha_n, b = self.beta_n, size = n)
        return random_state.geometric(p = p, size = n)
    
    def plot(self, plot_type): 
        x = np.linspace(0,1,100)
        if plot_type == 'prior':
            self._check_plot(['alpha_0','beta_0','alpha_n','beta_n'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_0, b = self.beta_0), label = 'Prior')
        
        if plot_type == 'posterior':
            self._check_plot(['alpha_n','beta_n'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_n, b = self.beta_n), label = 'Posterior')

        if plot_type == 'both':
            self._check_plot(['alpha_0','beta_0','alpha_n','beta_n'], plot_type)
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_0, b = self.beta_0), label = 'Prior')
            plt.plot(x,stats.beta.pdf(x, a = self.alpha_n, b = self.beta_n), label = 'Posterior')


        plt.xlabel('p')
        plt.ylabel('pdf')
        plt.legend()
        plt.title("Distribution of p")
        plt.show()
    
#TODO: finish hypergeometric 
class hypergeometric(model):

    def update_model(self, data, **params):
        super()._update_model(**params)
        self._check_params(['N','n', 'alpha_0','beta_0'])

        data = np.asarray(data)

        self.alpha_n = self.alpha_0 + np.sum(data,axis = 0)




    


    
