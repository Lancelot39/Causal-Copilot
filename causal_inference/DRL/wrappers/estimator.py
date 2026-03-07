import numpy as np
import pandas as pd
from typing import Dict, Tuple

from econml.dr import DRLearner as Econ_DRL
from econml.dr import LinearDRLearner as Econ_LinearDRL
from econml.dr import SparseLinearDRLearner as Econ_SparseLinearDRL
from econml.dr import ForestDRLearner as Econ_ForestDRL
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from .base import Estimator

# - DRL
# - LinearDRL
# - SparseLinearDRLL
# - ForestDRL

class DRL(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int=0, T1: int=1):
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        self.model = Econ_DRL(**self._params)


    @property
    def name(self):
        return "DRL"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col] if self.X_col else None
        W = data[self.W_col] if self.W_col else None
        # Run DRL algorithm        
        self.model.fit(Y=y, T=T, X=X, W=W)
            
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col] if self.X_col else None
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col] if self.X_col else None
        treated_indices = (data[self.T_col] == 1)
        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass


class LinearDRL(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int=0, T1: int=1):
        del params['model_final']
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        self.model = Econ_LinearDRL(cv=5, **self._params)

    @property
    def name(self):
        return "LinearDRL"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col] if self.X_col else None
        W = data[self.W_col] if self.W_col else None
        # Run DRL algorithm        
        self.model.fit(y, T, X=X, W=W)
        
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col] if self.X_col else None
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        treated_indices = np.isclose(data[self.T_col], self.T1)
        if treated_indices.sum() == 0:
            print(f"[WARN] No treated samples found for T1 = {self.T1}. ATT cannot be computed.")
            return np.nan, np.nan, np.nan

        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass

class SparseLinearDRL(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int=0, T1: int=1):
        del params['model_final']
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        self.model = Econ_SparseLinearDRL(cv=5, **self._params)

    @property
    def name(self):
        return "SparseLinearDRL"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col] if self.X_col else None
        W = data[self.W_col] if self.W_col else None
        # Run DRL algorithm        
        self.model.fit(y, T, X=X, W=W)
           
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col] if self.X_col else None
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
        treated_indices = np.isclose(data[self.T_col], self.T1)
        if treated_indices.sum() == 0:
            print(f"[WARN] No treated samples found for T1 = {self.T1}. ATT cannot be computed.")
            return np.nan, np.nan, np.nan
        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass

class ForestDRL(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int = 0, T1: int = 1):
        # del params['model_final']
        # super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        # self.model = Econ_ForestDRL(cv=5, **self._params)
        params.pop('model_final', None)

        # ✅ Safe CV logic
        try:
            # We'll assume the treatment is available at init time for now
            # You can patch this dynamically later if needed
            treatment_values = pd.Series(params.get("T_vals", []))  # You could pass T_vals when calling this
            min_treatment_count = treatment_values.value_counts().min()

            if min_treatment_count >= 3:
                print("[INFO] Using StratifiedKFold(n_splits=3)")
                params['cv'] = StratifiedKFold(n_splits=3)
            else:
                print("[WARN] Treatment group too small. Using KFold(n_splits=2)")
                params['cv'] = KFold(n_splits=2)
        except Exception:
            print("[WARN] Could not determine safe CV strategy. Using default KFold(n_splits=2)")
            params['cv'] = KFold(n_splits=2)

        # Now continue with standard init
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        # Fix the model_propensity if it's missing or incorrect
        if 'model_propensity' not in self._params or isinstance(self._params['model_propensity'], RandomForestRegressor):
            print("[FIX] Setting model_propensity to RandomForestClassifier")
            self._params['model_propensity'] = RandomForestClassifier(n_estimators=100)


        self.model = Econ_ForestDRL(**self._params)

    @property
    def name(self):
        return "CausalForestDRL"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[[self.T_col]]
        X = data[self.X_col] if self.X_col else None
        W = data[self.W_col] if self.W_col else None
        # Run DRL algorithm        
        self.model.fit(y, T, X=X, W=W)
    
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col] if self.X_col else None
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X,T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col] if self.X_col else None
        treated_indices = np.isclose(data[self.T_col], self.T1)
        if treated_indices.sum() == 0:
            print(f"[WARN] No treated samples found for T1 = {self.T1}. ATT cannot be computed.")
            return np.nan, np.nan, np.nan
        treated_effects = self.model.effect(X[treated_indices], T0=self.T0, T1=self.T1)
        lower_bound, upper_bound = self.model.effect_interval(X[treated_indices], T0=self.T0, T1=self.T1)
        att = np.mean(treated_effects)
        att_lower, att_upper = np.mean(lower_bound), np.mean(upper_bound)
        return att, att_lower, att_upper

    def hte(self, data: pd.DataFrame):
        X = data[self.X_col]
        hte = self.model.effect(X, T0=self.T0, T1=self.T1)
        hte_lower, hte_upper = self.model.effect_interval(X, T0=self.T0, T1=self.T1)
        return hte, hte_lower, hte_upper

    def test_algorithm(self):
        pass


if __name__ == "__main__":
    pc_algo = DRL(y_col='', T_col='', X_col='')
    pc_algo.test_algorithm() 