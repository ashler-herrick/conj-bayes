class model:
    def __init__(self,**params):
        self.__dict__.update(params)

    
    def _update_model(self, **params):
        self.__dict__.update(params)
    
    def _check_params(self, attr_list):
        for attr in attr_list:
            if not hasattr(self,attr):
                print(f"Parameter {attr} not found. Please add {attr} to the model.")

