import sys
import os

#==================================================
#define helper methods for testing
#==================================================
method_universe = ['describe','plot','posterior_mean'
                   ,'posterior_mode','sample_posterior'
                   ,'sample_posterior_predictive','sample_prior_predictive']
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


#add path so we dont get ModuleNotFoundError
def add_path():
    cur_dir = os.getcwd()
    print(sys.path)
    sys.path.append(cur_dir)