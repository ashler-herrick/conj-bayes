class model:
    def __init__(self,**params):
        self.__dict__.update(params)

    
    def _update_model(self, **params):
        self.__dict__.update(params)
    
    def _check_params(self, attr_list):
        for attr in attr_list:
            if not hasattr(self,attr):
                print(f"Parameter {attr} not found. Please add {attr} to the model.")
                return False
        return True
    
    def _check_plot(self, attr_list, plot_type):
        check = self._check_params(['alpha_0','beta_0','alpha_n','beta_n'])
        if check == False and plot_type == None:
            plot_type = 'prior'
        elif check == False and plot_type == None:
            plot_type = 'both'
        else:
            self.plot_type = plot_type

    def describe(self):
        print(self.__dict__)

