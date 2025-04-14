import numpy as np
import pandas as pd
from typing import Dict
from econml.metalearners import SLearner as SLearnerEstimator
from econml.metalearners import TLearner as TLearnerEstimator
from econml.metalearners import XLearner as XLearnerEstimator 
from econml.metalearners import DomainAdaptationLearner as DomainAdaptationLearnerEstimator
from .base import Estimator
from sklearn.linear_model import LinearRegression 
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from econml.inference import BootstrapInference



# - SLearner
# - TLearner
# - XLearner
# - DomainAdaptationLearner

class SLearner(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int=0, T1: int=1):
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        slearner_params = {
            'categories': self._params.get('categories', 'auto'),  # Default: 'auto' to use the unique sorted values,The first category will be treated as the control treatment.
            'allow_missing': self._params.get('allow_missing', False)  # Default: False
            }
        base_learner = params.get('model', LinearRegression())  
        self.model = SLearnerEstimator(overall_model=base_learner, **slearner_params)

    @property
    def name(self):
        return "SLearner"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[self.T_col].values.ravel()
        X = data[self.X_col]
        # Fit SLearner model
        self.model.fit(Y=y, T=T, X=X, inference=BootstrapInference(n_bootstrap_samples=100))
    
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1 )
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
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

class TLearner(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int=0, T1: int=1):
        super().__init__(params, y_col, T0, T1, T_col, X_col, W_col)
        tlearner_params = {
            'categories': self._params.get('categories', 'auto'),  # Default: 'auto' to use the unique sorted values,The first category will be treated as the control treatment.
            'allow_missing': self._params.get('allow_missing', False)  # Default: False
            }
        base_learner = params.get('model', LinearRegression()) 
        print("Expected Treatment Column:", self.T_col)
        self.model = TLearnerEstimator(models=base_learner, **tlearner_params)

    @property
    def name(self):
        return "TLearner"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        
        y = data[[self.y_col]].values.ravel()
        if self.T_col not in data.columns:
            raise KeyError(f"Treatment column '{self.T_col}' not found in the dataset. Available columns: {data.columns}")
        T = data[self.T_col].values.ravel()
        X = data[self.X_col]
        # Fit TLearner model  
        # ✅ Explicitly set BootstrapInference    
        self.model.fit(y, T, X=X, inference=BootstrapInference(n_bootstrap_samples=100))
        
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
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

class XLearner(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int=0, T1: int=1):
        del params['model_final']
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        xlearner_params = {
            'categories': self._params.get('categories', 'auto'),  # Default: 'auto' to use the unique sorted values,The first category will be treated as the control treatment.
            'allow_missing': self._params.get('allow_missing', False) # Default: False
        }
        # Use XGBoost as base learners for treatment and control groups
        base_learner = params.get('model', XGBRegressor(objective='reg:squarederror'))
        
        # Use XGBoost as the CATE model (same as base learners by default)
        cate_learner = params.get('cate_model', base_learner)
        
        # Use Logistic Regression as the propensity score model
        propensity_model = params.get('propensity_model', LogisticRegression())

        # Initialize XLearner
        self.model = XLearnerEstimator(
            models=base_learner, 
            cate_models=cate_learner, 
            propensity_model=propensity_model, 
            **xlearner_params
        )

    @property
    def name(self):
        return "XLearner"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[self.T_col].values.ravel()
        X = data[self.X_col]
        # Fit XLearner model        
        self.model.fit(y, T, X=X, inference=BootstrapInference(n_bootstrap_samples=100))
           
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
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

class DomainAdaptationLearner(Estimator):
    def __init__(self, y_col: str, T_col: str, X_col: list, params: Dict = {}, W_col: list = None, T0: int = 0, T1: int = 1):
        del params['model_final']
        super().__init__(params, y_col, T_col, T0, T1, X_col, W_col)
        # Ensure valid parameters
        dalearner_params = {
            'categories': self._params.get('categories', 'auto'),# Default: 'auto' to use the unique sorted values,The first category will be treated as the control treatment.
            'allow_missing': self._params.get('allow_missing', False)# Default: False
        }
        # Use XGBoost as the base model for treatment and control groups
        base_learner = params.get('model', XGBRegressor(objective='reg:squarederror'))

        # Use XGBoost as the final treatment effect model
        final_learner = params.get('final_model', base_learner)

        # Use Logistic Regression as the propensity score model
        propensity_model = params.get('propensity_model', LogisticRegression())

        self.model = DomainAdaptationLearnerEstimator(
            models=base_learner,
            final_models=final_learner,
            propensity_model=propensity_model,
            **dalearner_params
        )

    @property
    def name(self):
        return "DomainAdaptationLearner"

    def get_params(self):
        return self._params

    def fit(self, data: pd.DataFrame):
        y = data[[self.y_col]].values.ravel()
        T = data[self.T_col].values.ravel()
        X = data[self.X_col]
        # Fit DomainAdaptationLearner model  
        self.model.fit(y, T, X=X, inference=BootstrapInference(n_bootstrap_samples=100))
    
    def ate(self, data: pd.DataFrame):
        X = data[self.X_col]
        ate = self.model.ate(X=X, T0=self.T0, T1=self.T1)
        ate_lower, ate_upper = self.model.ate_interval(X=X, T0=self.T0, T1=self.T1)
        return ate, ate_lower, ate_upper

    def att(self, data: pd.DataFrame):
        X = data[self.X_col]
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


if __name__ == "__main__":
    pc_algo = DomainAdaptationLearner(y_col='', T_col='', X_col='')
    pc_algo.test_algorithm() 