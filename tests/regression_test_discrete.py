
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
#test binomial
#==================================================
#generating test data

test_data = np.random.randint(max_test, size = 100)
binom = conj_bayes.binomial(alpha_0 = 1, beta_0  = 1, m = max_test)
test_method_wrapper(test_data,binom)

#==================================================
#test bernoulli
#==================================================
test_data = np.random.randint(2,size = 100)
bern = conj_bayes.bernoulli(alpha_0 = 1, beta_0 = 1)
test_method_wrapper(test_data,bern)

#==================================================
#test negative_binomial
#==================================================

test_data = np.random.randint(max_test, size = 100)
nbin = conj_bayes.negative_binomial(alpha_0 = 1, beta_0 = 1, r = max_test)
test_method_wrapper(test_data,nbin)

#==================================================
#test geometric
#==================================================   
test_data = np.random.randint(max_test, size = 100)
geom = conj_bayes.geometric(alpha_0 = 1, beta_0 = 1)
test_method_wrapper(test_data, geom)


#==================================================
#test poisson
#==================================================
test_data = np.random.randint(max_test, size = 100)
pois = conj_bayes.poisson(alpha_0 = 1, beta_0 = 1)
test_method_wrapper(test_data,pois)

#==================================================
#test categorical
#==================================================
test_data = np.zeros((100,10))
for row in test_data:
    i = np.random.randint(max_test)
    row[i] = 1
cat = conj_bayes.categorical(alpha_0 = np.ones(10))
test_method_wrapper(test_data,cat)

#==================================================
#test multinomial
#==================================================   
test_data = np.random.multinomial(max_test, np.ones(10)/10, size = 100)
mult = conj_bayes.multinomial(alpha_0 = np.ones(10), n = max_test)
test_method_wrapper(test_data, mult)