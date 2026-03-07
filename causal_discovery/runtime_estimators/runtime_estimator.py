import ast
import math
import operator

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import json
import os
from typing import Dict, List, Any
from llm import LLMClient
import scipy.special
import requests


def _safe_eval_expr(expr: str, variables: dict) -> float:
    """
    Safely evaluate a mathematical expression with variables.
    Only allows: +, -, *, /, **, math functions (log, sqrt, exp, abs), and numeric literals.
    """
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    _FUNCTIONS = {
        'log': math.log,
        'log2': math.log2,
        'log10': math.log10,
        'sqrt': math.sqrt,
        'exp': math.exp,
        'abs': abs,
        'max': max,
        'min': min,
    }

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant: {node.value!r}")
        elif isinstance(node, ast.Name):
            name = node.id
            if name in variables:
                return float(variables[name])
            raise ValueError(f"Unknown variable: {name!r}")
        elif isinstance(node, ast.BinOp):
            op_func = _OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_func(_eval_node(node.left), _eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_func = _OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
            return op_func(_eval_node(node.operand))
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            func = _FUNCTIONS.get(node.func.id)
            if func is None:
                raise ValueError(f"Unknown function: {node.func.id!r}")
            args = [_eval_node(arg) for arg in node.args]
            return float(func(*args))
        else:
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {expr!r}") from e
    return _eval_node(tree)

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
        self.algorithm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Load complexity terms configuration from file
        with open(os.path.join(self.algorithm_dir, 'runtime_estimators', 'complexity_terms.json'), 'r') as f:
            all_configs = json.load(f)
        if algorithm_name in all_configs:
            self.complexity_config = all_configs[algorithm_name]
        else:
            # For a new algorithm not registered in the JSON file, handle it here.
            self.complexity_config = self._handle_new_algorithm(algorithm_name, all_configs)
            
        self.feature_names = [term['name'] for term in self.complexity_config['terms']]
        self.log_transform = self.complexity_config['log_transform']
        self.min_runtime = self.complexity_config['min_runtime']
        
        # Load or fit model
        self._initialize_model()
    
    def _initialize_model(self):
        """Load existing model or fit new one."""
        fit_path = os.path.join(self.algorithm_dir, 'context', 'benchmarking', 'acceleration_fit.json')
        if os.path.exists(fit_path):
            with open(fit_path, 'r') as f:
                fit_data = json.load(f)
                if self.algorithm_name in fit_data:
                    self._load_model(fit_data[self.algorithm_name])
                    return
                    
        data = self._load_data()
        self._fit_model(data)

    def _load_data(self):
         # If no saved coefficients, fit model
        data = json.load(open(os.path.join(self.algorithm_dir, 'context', 'benchmarking', 'merged_results.json'), 'r'))
        # Screen the data specifically for the algorithm and extract relevant fields
        filtered_data = []
        for d in data:
            if d['algorithm'] == self.algorithm_name and d['success']:
                filtered_data.append({
                    'variables': d['data_config']['n_nodes'],
                    'samples': d['data_config']['n_samples'], 
                    'time_cost': d['runtime']
                })

        df = pd.DataFrame(filtered_data)
        # Rename columns to match our parameter naming convention
        column_mapping = {
            'variables': 'p',
            'samples': 'N',
            'time_cost': 'time_cost'
        }
        df = df.rename(columns=column_mapping)

        # remove rows with time_cost > 1 hours
        df = df[df['time_cost'] < 3600]

        return df
    
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
            'density': params.get('density', 0.2),
            'N': params.get('N', 5000)  # Default N to N if n exists, otherwise 5000
        }
        
        for key, default_value in defaults.items():
            if key not in params:
                params[key] = default_value
        
        if 'param_calculations' in self.complexity_config:
            for param, calc in self.complexity_config['param_calculations'].items():
                try:
                    # Build variables dict for safe evaluation
                    eval_vars = {k: float(v) for k, v in params.items()
                                 if isinstance(v, (int, float, np.number))}

                    # Evaluate the expression safely
                    result = _safe_eval_expr(calc['expression'], eval_vars)
                    params[param] = float(result) if isinstance(result, (int, float, np.number)) else result
                    
                except Exception as e:
                    print(f"Warning: Failed to calculate {param}: {str(e)}")
                    params[param] = 0
                    
        return params
    
    def _evaluate_expression(self, expr: str, variables: Dict[str, float]) -> float:
        """Safely evaluate a complexity term expression."""
        eval_vars = {k: float(v) for k, v in variables.items()
                     if isinstance(v, (int, float, np.number))}
        return _safe_eval_expr(expr, eval_vars)
    
    def _calculate_features(self, params: Dict[str, float]) -> pd.Series:
        """Calculate feature values based on complexity terms."""
        # First calculate any derived parameters
        full_params = self._calculate_derived_params(params)
        
        features = {}
        for term in self.complexity_config['terms']:
            features[term['name']] = self._evaluate_expression(term['expression'], full_params)
        return pd.Series(features)
    
    def _fit_model(self, df: pd.DataFrame):
        """Fit the model using training data."""
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
        X_features = X_features.values
        
        # Transform target variable if needed
        y = np.log(df['time_cost']) if self.log_transform else df['time_cost']
        
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
        fit_path = os.path.join(self.algorithm_dir, 'context', 'benchmarking', 'acceleration_fit.json')
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
    
    def predict_runtime(self, truncate_runtime=True, **params) -> float:
        """Predict runtime using the complexity formula."""
        features = self._calculate_features(params)
        # Ensure feature order matches training data
        features = pd.DataFrame([features])[self.feature_names].values
        # Replace any infinite values with a large number to avoid prediction issues
        features = np.nan_to_num(features, posinf=1e10, neginf=-1e10)
        
        predicted = self.model.predict(features)[0]
        if self.log_transform:
            predicted = np.exp(predicted)

        # hard coded for now for CDNOD small sample case
        if predicted == np.inf:
            predicted = 0.002
            
        if truncate_runtime:
            return max(np.abs(predicted), self.min_runtime)
        else:
            return np.abs(predicted)
    
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
            
        predicted_time = self.predict_runtime(truncate_runtime=False, **params)
        print(f"\nPredicted runtime: {predicted_time:.3f} seconds")
    
    def test_accuracy(self, test_cases=None):
        """Test the model accuracy using provided test cases or default ones."""
        if test_cases is None:
            # Default test cases for each algorithm
            # Load test cases from acceleration.json
            df = self._load_data()
            
            # Convert data to list of test cases
            test_cases = []
            for i, row in df.iterrows():
                if row['time_cost'] < 43200:
                    case = {
                        'variables': row['p'],
                        'samples': row['N'],
                        'actual': row['time_cost']
                    }
                    if 'density' in row:
                        case['density'] = row['density']
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
            predicted = self.predict_runtime(truncate_runtime=False, **params)
            
            # Calculate relative error
            rel_error = abs(predicted - actual) / actual * 100
            total_rel_error += rel_error
            
            # Print results
            density = params.get('density', '-')
            print(f"{params['variables']}\t{params['samples']}\t{density}\t{actual:.3f}\t{predicted:.3f}\t{rel_error:.1f}%")
        
        avg_rel_error = total_rel_error / len(test_cases)
        print(f"\nAverage relative error: {avg_rel_error:.1f}%")
        
        return avg_rel_error

    def _handle_new_algorithm(self, algorithm_name: str, all_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle case when the algorithm's complexity term is not registered.
        Load the mapping table to find the main code file(s), read their content,
        combine with a prompt template, query the LLM to deduce the time complexity,
        and update the complexity_terms.json file.
        """
        mapping_path = os.path.join(self.algorithm_dir, 'runtime_estimators', 'algorithm_mappings.json')
        if not os.path.exists(mapping_path):
            raise FileNotFoundError("Algorithm mapping table not found at " + mapping_path)
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        
            
        if algorithm_name not in mapping:
            raise ValueError(f"Algorithm '{algorithm_name}' not found in the mapping table.")
            
        # Support a single file or a list of code files.
        code_files = mapping[algorithm_name]
        if not isinstance(code_files, list):
            code_files = [code_files]
            
        code_contents = ""
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for file_path in code_files:
            # Check if the path is a URL
            if file_path.startswith(('http://', 'https://')):
                try:
                    response = requests.get(file_path)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    code_contents += f"Here is the code for {algorithm_name}, code URL: {file_path}:\n\n" + response.text + "\n\n"
                except requests.exceptions.RequestException as e:
                    print(f"Warning: Failed to fetch code from URL {file_path}: {e}")
            else:
                # Handle local file path
                file_path = file_path.replace("[root_dir]", root_dir)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as code_file:
                        code_contents += f"Here is the code for {algorithm_name}, code file: {file_path}:\n\n" + code_file.read() + "\n\n"
                else:
                    print(f"Warning: Code file {file_path} not found.")
                
        # Load the prompt template used to ask the LLM.
        prompt_template_path = os.path.join(self.algorithm_dir, 'runtime_estimators', 'time_complexity_prompt.txt')
        if not os.path.exists(prompt_template_path):
            raise FileNotFoundError("Time complexity prompt template not found at " + prompt_template_path)
        with open(prompt_template_path, 'r') as pt:
            prompt_template = pt.read()
            
        # Format the prompt with the algorithm name and the combined code.
        full_prompt = prompt_template.replace("[algorithm_name]", algorithm_name).replace("[code]", code_contents)
        full_prompt = full_prompt.replace("[example_json]", json.dumps([v for k, v in all_configs.items()]))
        
        # Query LLM to determine the algorithm's complexity in our JSON format.
        complexity_info = self._query_llm(full_prompt)

        # Add the default minimum runtime and log transform.
        complexity_info['min_runtime'] = 60
        
        # Update the complexity_terms.json file with the new algorithm information.
        all_configs[algorithm_name] = complexity_info
        with open(os.path.join(self.algorithm_dir, 'runtime_estimators', 'complexity_terms.json'), 'w') as f:
            json.dump(all_configs, f, indent=4)
            
        return complexity_info

    def _query_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Stub for querying an LLM with the given prompt.
        In a production environment, this should call the LLM API.
        For now, we simulate a response.
        """
        # Use o3-mini model to analyze code complexity
        client = LLMClient()
        
        response = client.chat_completion(
            prompt=prompt,
            system_prompt="You are a helpful assistant for runtime estimation.",
            json_response=True
        )
        return response

if __name__ == "__main__":
    # Test all algorithms
    algorithms = ['PCParallel'] # 'AcceleratedPC', 'BAMB', 'GOLEM', 'GRaSP', 'IAMBnPC', 'InterIAMB', 'MBOR', 'NOTEARSLinear', 'AcceleratedLiNGAM', 'DirectLiNGAM', 'PC', 'GES', 'FGES', 'XGES', 'FCI', 'CDNOD']
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