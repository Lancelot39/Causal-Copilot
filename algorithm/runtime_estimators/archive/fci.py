import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import json

class FCI:
    # Time complexity: O(n^2 * 2^n)
    def __init__(self):
        # Load training data
        data = json.load(open('algorithm/context/benchmarking/acceleration.json', 'r'))['fci']
        
        self.feature_names = ['log_n2', 'log_exp_n']
        self._fit_model(data)
        
    def estimate_max_degree(self, variables, density=0.3):
        """Estimate maximum node degree based on variables and density."""
        # Maximum possible edges per node is (V-1)
        # With density factor, expected edges per node is density * (V-1)
        return int(density * (variables - 1))
        
    def _fit_model(self, data):
        """Fit the model using the complexity terms."""
        df = pd.DataFrame(data)
        
        # Create feature matrix
        X_features = pd.DataFrame()
        
        # For each row, calculate complexity terms based on theoretical complexity
        def calculate_features(row):
            n = row['variables']
            N = row.get('samples', 5000)  # Default sample size if not provided
            density = row.get('density', 0.3)  # Default density if not provided
            
            # Take log of each complexity term separately
            return pd.Series({
                'log_n2': np.log(n * n),  # log of n^2 term for variable pairs
                'log_exp_n': n * np.log(2)  # log of 2^n term for conditioning sets
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

    def predict_runtime(self, n_variables, density=0.3, n_samples=5000):
        """Predict FCI runtime using the theoretical complexity formula."""
        # Calculate log of each term separately
        features = pd.DataFrame({
            'log_n2': [np.log(n_variables * n_variables)],
            'log_exp_n': [n_variables * np.log(2)]
        })
        
        # Predict log time and convert back
        log_time = self.model.predict(features)[0]
        predicted_time = np.exp(log_time)
        
        # Ensure minimum runtime
        return max(predicted_time, 0.001)

    def explain_complexity(self, n_variables, density=0.3, n_samples=5000):
        """Explain the complexity breakdown for given dimensions."""
        n2_term = n_variables * n_variables
        exp_term = 2 ** n_variables
        
        print(f"\nComplexity Analysis for {n_variables} variables with density {density}:")
        print(f"n^2 term: {n2_term}")
        print(f"2^n term: {exp_term}")
        print(f"Sample term (N): {n_samples}")
        
        predicted_time = self.predict_runtime(n_variables, density, n_samples)
        print(f"Predicted runtime: {predicted_time:.3f} seconds")

if __name__ == "__main__":
    predictor = FCI()
    
    # Test cases from the training data
    test_cases = [
        (5, 500, 0.3, 0.037),
        (10, 500, 0.3, 0.001), 
        (20, 500, 0.3, 0.006),
        (50, 500, 0.3, 0.046),
        (100, 500, 0.3, 0.412),
        (200, 500, 0.3, 3.821),
        (500, 500, 0.3, 155.859),
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
