import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import json
import os

class CDNOD:
    # Time complexity: O((n+1)^(2+d) + (n+1)^4)
    def __init__(self):
        # Load training data
        data = json.load(open('algorithm/context/benchmarking/acceleration.json', 'r'))['cdnod']
        
        self.feature_names = ['log_n_2d', 'log_n4']
        self._fit_model(data)
        
    def estimate_max_degree(self, variables, density=0.3):
        """Estimate maximum node degree based on variables and density."""
        # Maximum possible edges per node is (V-1)
        # With density factor, expected edges per node is density * (V-1)
        return int(density * (variables - 1))
        
    def _fit_model(self, data):
        """Fit the model using the complexity terms."""
        # Check if fitted coefficients exist
        fit_path = 'algorithm/context/benchmarking/acceleration_fit.json'
        if os.path.exists(fit_path):
            with open(fit_path, 'r') as f:
                fit_data = json.load(f)
                if 'cdnod' in fit_data:
                    # Load existing coefficients
                    coef_data = fit_data['cdnod']
                    self.model = LinearRegression(fit_intercept=True)
                    self.model.coef_ = np.array(coef_data['coefficients'])
                    self.model.intercept_ = coef_data['intercept']
                    
                    # Print loaded coefficients
                    print("Loaded Model Coefficients:")
                    for name, coef in zip(self.feature_names, self.model.coef_):
                        print(f"{name}: {coef:.2e}")
                    print(f"intercept: {self.model.intercept_:.2e}")
                    
                    return

        # Original fitting code...
        df = pd.DataFrame(data)
        
        # Create feature matrix
        X_features = pd.DataFrame()
        
        # For each row, calculate complexity terms based on theoretical complexity
        def calculate_features(row):
            n = row['variables']
            N = row.get('samples', 5000)  # Default sample size if not provided
            density = row.get('density', 0.3)  # Default density if not provided
            d = self.estimate_max_degree(n, density)
            
            # Take log of each complexity term separately
            return pd.Series({
                'log_n_2d': (2 + d) * np.log(n + 1),  # log of (n+1)^(2+d) term
                'log_n4': 4 * np.log(n + 1)  # log of (n+1)^4 term
            })
        
        X_features = df.apply(calculate_features, axis=1)
        
        # Filter out timeout cases
        mask = df['time_cost'] < 43200
        X_features = X_features[mask]
        y = np.log(df['time_cost'][mask])  # Take log of time cost
        
        # Fit model
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(X_features, y)
        
        # Print coefficients
        print("Model Coefficients:")
        for name, coef in zip(self.feature_names, self.model.coef_):
            print(f"{name}: {coef:.2e}")
        print(f"intercept: {self.model.intercept_:.2e}")
        
        # Save coefficients
        fit_path = 'algorithm/context/benchmarking/acceleration_fit.json'
        if os.path.exists(fit_path):
            with open(fit_path, 'r') as f:
                fit_data = json.load(f)
        else:
            fit_data = {}
        
        fit_data['cdnod'] = {
            'coefficients': self.model.coef_.tolist(),
            'intercept': float(self.model.intercept_)
        }
        
        with open(fit_path, 'w') as f:
            json.dump(fit_data, f, indent=4)

    def predict_runtime(self, n_variables, density=0.3, n_samples=5000):
        """Predict CDNOD runtime using the theoretical complexity formula."""
        d = self.estimate_max_degree(n_variables, density)
        
        # Calculate log of each term separately
        features = pd.DataFrame({
            'log_n_2d': [(2 + d) * np.log(n_variables + 1)],
            'log_n4': [4 * np.log(n_variables + 1)]
        })
        
        # Predict log time and convert back
        log_time = self.model.predict(features)[0]
        predicted_time = np.exp(log_time)
        
        # Ensure minimum runtime
        return max(predicted_time, 0.001)

    def explain_complexity(self, n_variables, density=0.3, n_samples=5000):
        """Explain the complexity breakdown for given dimensions."""
        d = self.estimate_max_degree(n_variables, density)
        n_2d_term = (n_variables + 1) ** (2 + d)
        n4_term = (n_variables + 1) ** 4
        
        print(f"\nComplexity Analysis for {n_variables} variables with density {density}:")
        print(f"(n+1)^(2+d) term: {n_2d_term}")
        print(f"(n+1)^4 term: {n4_term}")
        print(f"Sample term (N): {n_samples}")
        
        predicted_time = self.predict_runtime(n_variables, density, n_samples)
        print(f"Predicted runtime: {predicted_time:.3f} seconds")

if __name__ == "__main__":
    predictor = CDNOD()
    
    # Test cases from the training data
    test_cases = [
        (5, 500, 0.3, 0.034),
        (10, 500, 0.3, 0.033),
        (20, 500, 0.3, 0.032),
        (50, 500, 0.3, 0.140),
        (100, 500, 0.3, 0.863),
        (200, 500, 0.3, 18.042),
        (500, 500, 0.3, 102.614),
    ]
    
    print("\nValidation against empirical data:")
    print("vars\tsamples\tdensity\tactual\tpredicted\trel_error")
    print("-" * 70)
    
    total_rel_error = 0
    for vars, samples, density, actual in test_cases:
        predicted = predictor.predict_runtime(vars, density)
        rel_error = abs(predicted - actual) / actual * 100
        total_rel_error += rel_error
        print(f"{vars}\t{samples}\t{density:.1f}\t{actual:.3f}\t{predicted:.3f}\t{rel_error:.1f}%")
    
    print(f"\nAverage relative error: {total_rel_error/len(test_cases):.1f}%")
