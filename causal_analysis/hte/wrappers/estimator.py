import numpy as np
import pandas as pd
from typing import Dict, Tuple

from econml.dml import DML as Econ_DML
from econml.dml import LinearDML as Econ_LinearDML
from econml.dml import SparseLinearDML as Econ_SparseLinearDML
from econml.dml import CausalForestDML as Econ_CausalForestDML

from .base import Estimator

# - DML 
# - LinearDML
# - SparseLinearDML
# - CausalForestDML
# - metalearners

class DML(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None):
        super().__init__(params, y_col, T_col, X_col, W_col)
        self.model = Econ_DML(**self._params)

    @property
    def name(self):
        return "DML"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]]
        T = data[[self.T_col]]
        X = data[self.X_col]
        W = data[self.W_col]
        # Run DML algorithm        
        self.model.fit(y, T, X=X, W=W)

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X)
        hte_lower, hte_upper = self.model.effect_interval(X)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass

class LinearDML(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None):
        del params['model_final']
        super().__init__(params, y_col, T_col, X_col, W_col)
        self.model = Econ_LinearDML(**self._params)

    @property
    def name(self):
        return "LinearDML"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]]
        T = data[[self.T_col]]
        X = data[self.X_col]
        W = data[self.W_col]
        # Run DML algorithm        
        self.model.fit(y, T, X=X, W=W)

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X)
        hte_lower, hte_upper = self.model.effect_interval(X)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass

class SparseLinearDML(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None):
        del params['model_final']
        super().__init__(params, y_col, T_col, X_col, W_col)
        self.model = Econ_SparseLinearDML(**self._params)

    @property
    def name(self):
        return "SparseLinearDML"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]]
        T = data[[self.T_col]]
        X = data[self.X_col]
        W = data[self.W_col]
        # Run DML algorithm        
        self.model.fit(y, T, X=X, W=W)

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X)
        hte_lower, hte_upper = self.model.effect_interval(X)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass

class CausalForestDML(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None):
        del params['model_final']
        super().__init__(params, y_col, T_col, X_col, W_col)
        self.model = Econ_CausalForestDML(**self._params)

    @property
    def name(self):
        return "CausalForestDML"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]]
        T = data[[self.T_col]]
        X = data[self.X_col]
        W = data[self.W_col]
        # Run DML algorithm        
        self.model.fit(y, T, X=X, W=W)

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X)
        hte_lower, hte_upper = self.model.effect_interval(X)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass


if __name__ == "__main__":
    pc_algo = DML(y_col='', T_col='', X_col='')
    pc_algo.test_algorithm() 