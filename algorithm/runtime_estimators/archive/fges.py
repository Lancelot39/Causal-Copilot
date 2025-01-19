import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import json
import os

class FGES:
    # Time complexity: O(p^2 * 2^p) for dense graphs
    # For sparse graphs with max degree q: O(p^q)
    def __init__(self):
        # Training data
        data = json.load(open('algorithm/context/benchmarking/acceleration.json', 'r'))['fges']
        
        self.feature_names = ['p2_term', 'exp_p_term', 'score_term', 'iteration_term']
        self._fit_model(data)
        
    def estimate_edges(self, variables, density=0.3):
        """Estimate number of edges based on variables and density."""
        # Maximum possible edges in a DAG is V(V-1)/2
        max_edges = (variables * (variables - 1)) / 2
        return int(max_edges * density)
        
    def _fit_model(self, data):
        """Fit the model using the complexity terms."""
        # Check if fitted coefficients exist
        fit_path = 'algorithm/context/benchmarking/acceleration_fit.json'
        if os.path.exists(fit_path):
            with open(fit_path, 'r') as f:
                fit_data = json.load(f)
                if 'fges' in fit_data:
                    # Load existing coefficients
                    coef_data = fit_data['fges']
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
            V = row['variables']
            N = row.get('samples', 5000)  # Default sample size if not provided
            density = row.get('density', 0.3)  # Default density if not provided
            E = self.estimate_edges(V, density)
            
            return pd.Series({
                'p2_term': np.log(V * V),  # log of p^2 term for node pairs
                'exp_p_term': np.log(2 ** V),  # log of 2^p term for neighbor subsets
                'score_term': np.log(N),  # log of score complexity
                'iteration_term': np.log(E)  # log of iterations term
            })
        
        X_features = df.apply(calculate_features, axis=1)
        
        # Filter out timeout cases (where time_cost >= 43200)
        mask = df['time_cost'] < 43200
        X_features = X_features[mask]
        y = np.log(df['time_cost'][mask])  # Take log of target variable
        
        # Fit model
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(X_features, y)
        
        # Print coefficients
        print("Model Coefficients (for log-transformed features):")
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
        
        fit_data['fges'] = {
            'coefficients': self.model.coef_.tolist(),
            'intercept': float(self.model.intercept_)
        }
        
        with open(fit_path, 'w') as f:
            json.dump(fit_data, f, indent=4)

    def predict_runtime(self, n_variables, density=0.3, n_samples=5000):
        """Predict FGES runtime using the theoretical complexity formula."""
        E = self.estimate_edges(n_variables, density)
        
        # Create features based on theoretical complexity terms
        features = pd.DataFrame({
            'p2_term': [np.log(n_variables * n_variables)],
            'exp_p_term': [np.log(2 ** n_variables)],
            'score_term': [np.log(n_samples)],
            'iteration_term': [np.log(E)]
        })
        
        # Predict log time and transform back
        log_prediction = self.model.predict(features)[0]
        return max(np.exp(log_prediction), 0.01)

    def explain_complexity(self, n_variables, density=0.3, n_samples=5000):
        """Explain the complexity breakdown for given dimensions."""
        E = self.estimate_edges(n_variables, density)
        
        print(f"\nComplexity Analysis for {n_variables} variables with density {density}:")
        print(f"Estimated edges (E): {E}")
        print(f"log(p²) term: {np.log(n_variables * n_variables):.3f}")
        print(f"log(2^p) term: {np.log(2 ** n_variables):.3f}")
        print(f"log(score term): {np.log(n_samples):.3f}")
        print(f"log(iteration term): {np.log(E):.3f}")
        
        predicted_time = self.predict_runtime(n_variables, density, n_samples)
        print(f"Predicted runtime: {predicted_time:.3f} seconds")


if __name__ == "__main__":
    predictor = FGES()
    
    # Test cases from the training data
    test_cases = [
        (5, 5000, 0.2, 0.01),
        (25, 5000, 0.2, 2.4),
        (25, 5000, 0.1, 0.13),
        (25, 5000, 0.3, 14.04),
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
    
    # # Analyze some interesting cases
    # predictor.explain_complexity(25, 0.2)
    # predictor.explain_complexity(50, 0.2)
    # predictor.explain_complexity(25, 0.3)


