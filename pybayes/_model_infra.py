import matplotlib.pyplot as plt

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
        #check if parameter is missing
        check = self._check_params(attr_list)
        #if we are missing a parameter and no plot was specified plot the prior
        if check == False and plot_type == None:
            self.plot_type = 'prior'
        #if we have all parameters then plot both
        elif check == True and plot_type == None:
            self.plot_type = 'both'
        #otherwise plot the specified 
        else:
            self.plot_type = plot_type

    def _show_plot(self, xlab, ylab):
        title = f'Distribution of {xlab}'
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.title(title)
        plt.show()

    #TODO: make the describe method look nicer. maybe print a dataframe
    def describe(self):
        print(self.__dict__)

