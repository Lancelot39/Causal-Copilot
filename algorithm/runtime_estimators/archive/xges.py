import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import json

class XGES:
    # Time complexity: O(p^2 * 2^k * score_complexity * number_of_iterations)
    # Where:
    # p = number of variables 
    # k = max number of neighbors for any node (related to graph density)
    # score_complexity = O(n) where n is number of samples
    # number_of_iterations = O(E) where E is number of edges
    
    def __init__(self):
        # Load training data
        data = json.load(open('algorithm/context/benchmarking/acceleration.json', 'r'))['xges']
        
        self.feature_names = ['log_p2', 'log_exp_k', 'log_score', 'log_iter']
        self._fit_model(data)

    def estimate_edges(self, variables, density=0.3):
        """Estimate number of edges based on variables and density."""
        # Maximum possible edges in a DAG is V(V-1)/2
        max_edges = (variables * (variables - 1)) / 2
        return int(max_edges * density)

    def _fit_model(self, data):
        """Fit the model using the complexity terms."""
        df = pd.DataFrame(data)
        
        # Create feature matrix
        X_features = pd.DataFrame()
        
        # For each row, calculate complexity terms based on theoretical complexity
        def calculate_features(row):
            V = row['variables']
            N = row.get('samples', 5000)  # Default sample size if not provided
            density = row.get('density', 0.3)  # Default density if not provided
            E = self.estimate_edges(V, density)
            k = int(np.log2(E)) if E > 0 else 0  # Estimate k based on edges
            
            # Take log of each complexity term separately
            return pd.Series({
                'log_p2': np.log(V * V),  # log of p^2 term
                'log_exp_k': k * np.log(2),  # log of 2^k term
                'log_score': np.log(N),  # log of score complexity
                'log_iter': np.log(E) if E > 0 else 0  # log of iterations
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
        """Predict XGES runtime using the theoretical complexity formula."""
        E = self.estimate_edges(n_variables, density)
        k = int(np.log2(E)) if E > 0 else 0
        
        # Calculate log of each term separately
        features = pd.DataFrame({
            'log_p2': [np.log(n_variables * n_variables)],
            'log_exp_k': [k * np.log(2)],
            'log_score': [np.log(n_samples)],
            'log_iter': [np.log(E) if E > 0 else 0]
        })
        
        # Predict log time and convert back
        log_time = self.model.predict(features)[0]
        predicted_time = np.exp(log_time)
        
        # Ensure minimum runtime
        return max(predicted_time, 0.01)

    def explain_complexity(self, n_variables, density=0.3, n_samples=5000):
        """Explain the complexity breakdown for given dimensions."""
        E = self.estimate_edges(n_variables, density)
        k = int(np.log2(E)) if E > 0 else 0
        
        print(f"\nComplexity Analysis for {n_variables} variables with density {density}:")
        print(f"Estimated edges (E): {E}")
        print(f"Estimated k: {k}")
        print(f"p² term: {n_variables * n_variables}")
        print(f"2^k term: {2 ** k}")
        print(f"Score term (n_samples): {n_samples}")
        print(f"Iteration term (E): {E}")
        
        predicted_time = self.predict_runtime(n_variables, density, n_samples)
        print(f"Predicted runtime: {predicted_time:.3f} seconds")

if __name__ == "__main__":
    predictor = XGES()
    
    # Test cases from the training data
    test_cases = [
        (5, 5000, 0.2, 17.23),
        (25, 5000, 0.2, 21.2),
        (25, 5000, 0.1, 16.17),
        (25, 5000, 0.3, 57.21),
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
    
    # Analyze some interesting cases
    predictor.explain_complexity(25, 0.2)
    predictor.explain_complexity(50, 0.2)
    predictor.explain_complexity(25, 0.3)


