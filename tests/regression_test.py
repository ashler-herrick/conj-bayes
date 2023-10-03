import sys
import os
cur_dir = os.getcwd()
print(sys.path)
sys.path.append(cur_dir)

import conj_bayes
import numpy as np

#set random seed
np.random.seed(64)
max_test = 10
method_universe = ['describe','plot','posterior_mean'
                   ,'posterior_mode','sample_posterior'
                   ,'sample_posterior_predictive','sample_prior_predictive']

#==================================================
#define helper methods for testing
#==================================================
#method for running all class methods
def test_methods(obj):
    # Get all the methods of the class
    method_names = [method for method in dir(obj) if callable(getattr(obj, method))]

    # Filter out methods not in the universe
    method_names = [method for method in method_names if method in method_universe]

    # Run all the methods
    for method_name in method_names:
        print(f'Running {method_name}')

        method = getattr(obj, method_name)
        if method_name == 'plot':
            method(plot_type = 'both')
        else:
            method()
#method for print formatting
def test_method_wrapper(test_data, obj):
    print(f'Testing {obj.__class__.__name__} class')
    obj.update_model(test_data)
    test_methods(obj)
    print(f'Done testing {obj.__class__.__name__} \n\n')
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