import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import json
import os
from typing import Dict, List, Any

# Unified time complexity units
# N: number of samples
# p: number of variables
# M: rank of the matrix
# d: degree of the graph
# E: number of edges

class RuntimeEstimator:
    def __init__(self, algorithm_name: str):
        """Initialize estimator for a specific algorithm."""
        self.algorithm_name = algorithm_name
        
        # Load complexity terms configuration
        with open('algorithm/runtime_estimators/complexity_terms.json', 'r') as f:
            self.complexity_config = json.load(f)[algorithm_name]
            
        self.feature_names = [term['name'] for term in self.complexity_config['terms']]
        self.log_transform = self.complexity_config['log_transform']
        self.min_runtime = self.complexity_config['min_runtime']
        
        # Load or fit model
        self._initialize_model()
    
    def _initialize_model(self):
        """Load existing model or fit new one."""
        fit_path = 'algorithm/context/benchmarking/acceleration_fit.json'
        if os.path.exists(fit_path):
            with open(fit_path, 'r') as f:
                fit_data = json.load(f)
                if self.algorithm_name in fit_data:
                    self._load_model(fit_data[self.algorithm_name])
                    return
                    
        # If no saved coefficients, fit model
        data = json.load(open('algorithm/context/benchmarking/acceleration.json', 'r'))[self.algorithm_name]
        self._fit_model(data)
    
    def _load_model(self, coef_data: Dict[str, Any]):
        """Load model from saved coefficients."""
        self.model = LinearRegression(fit_intercept=True)
        self.model.coef_ = np.array(coef_data['coefficients'])
        self.model.intercept_ = coef_data['intercept']
        
        # print("Loaded Model Coefficients:")
        # for name, coef in zip(self.feature_names, self.model.coef_):
        #     print(f"{name}: {coef:.2e}")
        # print(f"intercept: {self.model.intercept_:.2e}")
    
    def _calculate_derived_params(self, base_params: Dict[str, float]) -> Dict[str, float]:
        """Calculate additional parameters based on configuration rules."""
        params = base_params.copy()
        
        # Ensure consistent parameter naming
        if 'variables' in params:
            params['p'] = params.pop('variables')
        if 'samples' in params:
            params['N'] = params.pop('samples')
        
        # Set default values for missing parameters
        defaults = {
            'density': params.get('density', 0.3),
            'N': params.get('N', 5000)  # Default N to N if n exists, otherwise 5000
        }
        
        for key, default_value in defaults.items():
            if key not in params:
                params[key] = default_value
        
        if 'param_calculations' in self.complexity_config:
            for param, calc in self.complexity_config['param_calculations'].items():
                try:
                    # Create a safe evaluation environment
                    safe_dict = {
                        'log': np.log,  # Use natural log
                        'int': int,
                        'min': min,
                        'max': max,
                        **{k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in params.items()}
                    }
                    
                    # Evaluate the expression
                    result = eval(calc['expression'], {"__builtins__": {}}, safe_dict)
                    params[param] = float(result) if isinstance(result, (int, float, np.number)) else result
                    
                except Exception as e:
                    print(f"Warning: Failed to calculate {param}: {str(e)}")
                    params[param] = 0
                    
        return params
    
    def _evaluate_expression(self, expr: str, variables: Dict[str, float]) -> float:
        """Safely evaluate a complexity term expression."""
        # Replace variable names with their values
        for var, value in variables.items():
            expr = expr.replace(var, str(value))
            
        # Define safe math functions
        safe_dict = {
            'log': np.log,
            'exp': np.exp,
            'pow': pow,
            'sqrt': np.sqrt,
            **{k: float(v) for k, v in variables.items()}
        }
        
        return eval(expr, {"__builtins__": {}}, safe_dict)
    
    def _calculate_features(self, params: Dict[str, float]) -> pd.Series:
        """Calculate feature values based on complexity terms."""
        # First calculate any derived parameters
        full_params = self._calculate_derived_params(params)
        
        features = {}
        for term in self.complexity_config['terms']:
            features[term['name']] = self._evaluate_expression(term['expression'], full_params)
        return pd.Series(features)
    
    def _fit_model(self, data: List[Dict[str, Any]]):
        """Fit the model using training data."""
        df = pd.DataFrame(data)
        
        # Rename columns to match our parameter naming convention
        column_mapping = {
            'variables': 'p',
            'samples': 'N',
            'time_cost': 'time_cost'
        }
        df = df.rename(columns=column_mapping)
        
        # Calculate features for each data point
        feature_list = []
        for _, row in df.iterrows():
            try:
                features = self._calculate_features(dict(row))
                feature_list.append(features)
            except Exception as e:
                print(f"Warning: Failed to calculate features for row: {e}")
                continue
        
        X_features = pd.DataFrame(feature_list, columns=self.feature_names)
        
        # Filter out timeout cases if needed
        mask = df['time_cost'] < 43200  # 12 hours
        X_features = X_features[mask].values
        
        # Transform target variable if needed
        y = np.log(df['time_cost'][mask]) if self.log_transform else df['time_cost'][mask]
        
        # Fit model
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(X_features, y)
        
        # Save coefficients
        self._save_coefficients()
        
        # Print coefficients
        print("Model Coefficients:")
        for name, coef in zip(self.feature_names, self.model.coef_):
            print(f"{name}: {coef:.2e}")
        print(f"intercept: {self.model.intercept_:.2e}")
    
    def _save_coefficients(self):
        """Save model coefficients to file."""
        fit_path = 'algorithm/context/benchmarking/acceleration_fit.json'
        if os.path.exists(fit_path):
            with open(fit_path, 'r') as f:
                fit_data = json.load(f)
        else:
            fit_data = {}
            
        fit_data[self.algorithm_name] = {
            'coefficients': self.model.coef_.tolist(),
            'intercept': float(self.model.intercept_)
        }
        
        with open(fit_path, 'w') as f:
            json.dump(fit_data, f, indent=4)
    
    def predict_runtime(self, **params) -> float:
        """Predict runtime using the complexity formula."""
        features = self._calculate_features(params)
        # Ensure feature order matches training data
        features = pd.DataFrame([features])[self.feature_names].values
        
        predicted = self.model.predict(features)[0]
        if self.log_transform:
            predicted = np.exp(predicted)
            
        return max(predicted, self.min_runtime)
    
    def explain_complexity(self, **params):
        """Explain the complexity breakdown for given dimensions."""
        full_params = self._calculate_derived_params(params)
        features = self._calculate_features(params)
        
        print(f"\nComplexity Analysis for {self.algorithm_name}:")
        print("\nParameters:")
        for param, value in full_params.items():
            print(f"{param}: {value}")
            
        print("\nComplexity Terms:")
        for term, value in features.items():
            print(f"{term}: {value:.2e}")
            
        predicted_time = self.predict_runtime(**params)
        print(f"\nPredicted runtime: {predicted_time:.3f} seconds")
    
    def test_accuracy(self, test_cases=None):
        """Test the model accuracy using provided test cases or default ones."""
        if test_cases is None:
            # Default test cases for each algorithm
            # Load test cases from acceleration.json
            with open('algorithm/context/benchmarking/acceleration.json', 'r') as f:
                data = json.load(f)[self.algorithm_name]
            
            # Convert data to list of test cases
            test_cases = []
            for i in range(len(data['variables'])):
                if data['time_cost'][i] < 43200:
                    case = {
                        'variables': data['variables'][i],
                        'samples': data['samples'][i],
                        'actual': data['time_cost'][i]
                    }
                    if 'density' in data:
                        case['density'] = data['density'][i]
                    test_cases.append(case)
        
        print(f"\nValidation against empirical data for {self.algorithm_name}:")
        print("vars\tsamples\tdensity\tactual\tpredicted\trel_error")
        print("-" * 70)
        
        total_rel_error = 0
        for case in test_cases:
            # Extract test parameters
            params = {k: v for k, v in case.items() if k != 'actual'}
            actual = case['actual']
            
            # Make prediction
            predicted = self.predict_runtime(**params)
            
            # Calculate relative error
            rel_error = abs(predicted - actual) / actual * 100
            total_rel_error += rel_error
            
            # Print results
            density = params.get('density', '-')
            print(f"{params['variables']}\t{params['samples']}\t{density}\t{actual:.3f}\t{predicted:.3f}\t{rel_error:.1f}%")
        
        avg_rel_error = total_rel_error / len(test_cases)
        print(f"\nAverage relative error: {avg_rel_error:.1f}%")
        
        return avg_rel_error

if __name__ == "__main__":
    # Test all algorithms
    algorithms = ['direct_lingam', 'accelerated_lingam', 'pc', 'ges', 'fges', 'xges', 'fci', 'cdnod', 'notears']
    overall_error = 0
    
    for algo in algorithms:
        print(f"\n{'='*50}")
        print(f"Testing {algo}")
        print('='*50)
        
        estimator = RuntimeEstimator(algo)
        avg_error = estimator.test_accuracy()
        overall_error += avg_error
    
    print(f"\n{'='*50}")
    print(f"Overall average relative error across all algorithms: {overall_error/len(algorithms):.1f}%") 