import json
import causal_analysis.DML.wrappers as wrappers
#import hte.wrappers as wrappers

class HTE_Programming(object):
    def __init__(self, args, y_col: str, T_col: str,  T0: int, T1: int, X_col: list, W_col: list=None):
        self.args = args
        self.y_col = y_col
        self.T_col = T_col
        self.T0 = T0
        self.T1 = T1
        self.X_col = X_col
        self.W_col = W_col
        self.model = None

    def fit_model(self, global_state):
        algo_func = getattr(wrappers, global_state.inference.hte_algo_json['name'])
        self.model = algo_func(params=global_state.inference.hte_model_param, 
                          y_col=self.y_col, T_col=self.T_col, X_col=self.X_col, W_col=self.W_col,
                          T0=self.T0, T1=self.T1)
        self.model.fit(global_state.user_data.processed_data)
    
    def forward(self, global_state, task='hte'): 
        if task == 'ate':
            try:
                ate, ate_lower, ate_upper = self.model.ate(global_state.user_data.processed_data) 
            except:
                print("ATE calculation failed. Please check the model and data.")
                ate, ate_lower, ate_upper =  None, None, None
            return ate, ate_lower, ate_upper
        elif task == 'att':
            try:
                att, att_lower, att_upper = self.model.att(global_state.user_data.processed_data)
            except:
                print("ATT calculation failed. Please check the model and data.")
                att, att_lower, att_upper =  None, None, None
            return att, att_lower, att_upper          
        elif task == 'hte':
            try:
                hte, hte_lower, hte_upper = self.model.hte(global_state.user_data.processed_data)
            except:
                print("HTE calculation failed. Please check the model and data.")
                hte, hte_lower, hte_upper =  None, None, None
            return hte, hte_lower, hte_upper
