import numpy as np

# class for multivariate normal likelihood using multivariate normal prior
# assumes known covariance
class multivariate_normal:
    def __init__(self):
        #known covariance matrix
        self.Sigma = None
        #mean vector
        self.mu_n = None
        #number of dimensions
        self.d = None

    #update the parameters of the distribution
    def update_params(self, data, mu_0, Sigma_0, Sigma):
    
        #make sure the data is a numpy array
        data = np.array(data)
        n = len(data)

        self.d = len(data[0])

        x_bar = np.mean(data,axis = 0)

        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_0_inv = np.linalg.inv(Sigma_0)

        #Sigma_n is the covariance matrix of the distribution of mean
        self.Sigma_n = np.linalg.inv(Sigma_0_inv + n * Sigma_inv)

        self.mu_n = self.Sigma_n @ (n * Sigma_inv @ x_bar + Sigma_0_inv @ mu_0)
