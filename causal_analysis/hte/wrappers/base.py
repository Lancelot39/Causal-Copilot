import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple

class Estimator:
    def __init__(self, params: Dict, y_col: str, T_col: str, X_col: list, W_col: list=None):
        self._params = {}
        self._params.update(params)
        self.model = None 
        self.y_col = y_col
        self.T_col = T_col
        self.X_col = X_col
        self.W_col = W_col

    def get_params(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    # def get_primary_params(self):
    #     raise NotImplementedError("This method should be implemented by subclasses")
    
    # def get_secondary_params(self):
    #     raise NotImplementedError("This method should be implemented by subclasses")

    def fit(self, data: pd.DataFrame):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def ate(self, data: pd.DataFrame):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def att(self, data: pd.DataFrame):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def cate(self, data: pd.DataFrame):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def hte(self, data: pd.DataFrame):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def test_algorithm(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    @property
    def name(self):
        raise NotImplementedError("This property should be implemented by subclasses")