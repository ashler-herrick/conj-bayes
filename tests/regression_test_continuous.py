import numpy as np
from _test_infra import *

#set random seed
np.random.seed(64)
max_test = 10
#add path for testing
add_path()
#now that weve added path import module
import conj_bayes

#==================================================
#test normal
#=================================================
test_data = np.random.normal(0,1,size = 100)
norm = conj_bayes.normal(mu_0 = 0, sigma_0 = 1, sigma = 1, prior = 'normal')
test_method_wrapper(test_data,norm)
norm = conj_bayes.normal(mu_0 = 0, sigma_0 = 1, kappa_0 = 1, nu_0 = 1, prior = 'norm_inv_chi_sq')
test_method_wrapper(test_data,norm)