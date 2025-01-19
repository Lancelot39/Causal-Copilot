import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import json

class NOTEARS:
    # Time complexity: O(n * p^2 * T)
    # n: number of samples
    # p: number of variables
    # T: number of gradient descent iterations (typically scales with p)
    def __init__(self):
        # Load training data
        data = json.load(open('algorithm/context/benchmarking/acceleration.json', 'r'))['notears']
        
        self.feature_names = ['log_n', 'log_p2', 'log_iter']
        self._fit_model(data)
        
    def _fit_model(self, data):
        """Fit the model using the complexity terms."""
        df = pd.DataFrame(data)
        
        # Create feature matrix
        X_features = pd.DataFrame()
        
        # For each row, calculate complexity terms based on theoretical complexity
        def calculate_features(row):
            p = row['variables']
            n = row.get('samples', 500)  # Default sample size if not provided
            
            # Take log of each complexity term separately
            return pd.Series({
                'log_n': np.log(n),  # log of sample size term
                'log_p2': np.log(p * p),  # log of p^2 term for matrix operations
                'log_iter': np.log(p)  # log of iteration term (scales with variables)
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

    def predict_runtime(self, n_variables, n_samples=500):
        """Predict NOTEARS runtime using the theoretical complexity formula."""
        # Calculate log of each term separately
        features = pd.DataFrame({
            'log_n': [np.log(n_samples)],
            'log_p2': [np.log(n_variables * n_variables)],
            'log_iter': [np.log(n_variables)]
        })
        
        # Predict log time and convert back
        log_time = self.model.predict(features)[0]
        predicted_time = np.exp(log_time)
        
        # Ensure minimum runtime
        return max(predicted_time, 0.01)

    def explain_complexity(self, n_variables, n_samples=500):
        """Explain the complexity breakdown for given dimensions."""
        print(f"\nComplexity Analysis for {n_variables} variables:")
        print(f"Sample size term (n): {n_samples}")
        print(f"Variable squared term (p²): {n_variables * n_variables}")
        print(f"Iteration term (T~p): {n_variables}")
        
        predicted_time = self.predict_runtime(n_variables, n_samples)
        print(f"Predicted runtime: {predicted_time:.3f} seconds")

if __name__ == "__main__":
    predictor = NOTEARS()
    
    # Test cases from the training data
    test_cases = [
        (5, 500, 0.031),
        (10, 500, 0.002),
        (20, 500, 0.010),
        (50, 500, 0.312),
        (100, 500, 23.633),
        (200, 500, 148.615),
        (500, 500, 782.693)
    ]
    
    print("\nValidation against empirical data:")
    print("vars\tsamples\tactual\tpredicted\trel_error")
    print("-" * 60)
    
    total_rel_error = 0
    for vars, samples, actual in test_cases:
        predicted = predictor.predict_runtime(vars, samples)
        rel_error = abs(predicted - actual) / actual * 100
        total_rel_error += rel_error
        print(f"{vars}\t{samples}\t{actual:.3f}\t{predicted:.3f}\t{rel_error:.1f}%")
    
    print(f"\nAverage relative error: {total_rel_error/len(test_cases):.1f}%")