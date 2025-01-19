import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import json
import os

class DirectLiNGAM:
    # time complexity: O(np3M2 + p4M3)
    # https://www.cs.helsinki.fi/u/ahyvarin/papers/Shimizu11JMLR.pdf
    def __init__(self):
        # Training data
        self.feature_names = ['np3m2_term', 'p4m3_term']
        
        # Check if fitted coefficients exist
        fit_path = 'algorithm/context/benchmarking/acceleration_fit.json'
        if os.path.exists(fit_path):
            with open(fit_path, 'r') as f:
                fit_data = json.load(f)
                if 'direct_lingam' in fit_data:
                    # Load existing coefficients
                    coef_data = fit_data['direct_lingam']
                    self.cpu_model = LinearRegression(fit_intercept=True)
                    self.cpu_model.coef_ = np.array(coef_data['coefficients'])
                    self.cpu_model.intercept_ = coef_data['intercept']
                    
                    # Print loaded coefficients
                    print("Loaded Model Coefficients:")
                    print("\nCPU Model:")
                    for name, coef in zip(self.feature_names, self.cpu_model.coef_):
                        print(f"{name}: {coef:.2e}")
                    print(f"intercept: {self.cpu_model.intercept_:.2e}")
                    
                    return
        # If no saved coefficients, fit model
        data = json.load(open('algorithm/context/benchmarking/acceleration.json', 'r'))['direct_lingam']
        self._fit_model(data)
        
    def estimate_rank_M(self, n, p):
        """Estimate maximal rank M based on data dimensions."""
        return max(int(np.log2(min(n, p))), 2)
        
    def _fit_model(self, data):
        """Fit the CPU and GPU models with consistent feature names."""
        df = pd.DataFrame(data)
        df['M'] = df.apply(lambda row: self.estimate_rank_M(row['samples'], row['variables']), axis=1)
        
        # Create feature matrix
        X_features = pd.DataFrame({
            self.feature_names[0]: df.apply(lambda row: row['samples'] * (row['variables']**3) * (row['M']**2), axis=1),
            self.feature_names[1]: df.apply(lambda row: row['variables']**4 * row['M']**3, axis=1)
        })
        
        # Fit models
        self.cpu_model = LinearRegression(fit_intercept=True)
        self.cpu_model.fit(X_features, df['time_cost'])
        
        # Print coefficients
        print("Model Coefficients:")
        print("\nCPU Model:")
        for name, coef in zip(self.feature_names, self.cpu_model.coef_):
            print(f"{name}: {coef:.2e}")
        print(f"intercept: {self.cpu_model.intercept_:.2e}")
        
        # Save coefficients
        fit_path = 'algorithm/context/benchmarking/acceleration_fit.json'
        if os.path.exists(fit_path):
            with open(fit_path, 'r') as f:
                fit_data = json.load(f)
        else:
            fit_data = {}
            
        fit_data['direct_lingam'] = {
            'coefficients': self.cpu_model.coef_.tolist(),
            'intercept': float(self.cpu_model.intercept_)
        }
        
        with open(fit_path, 'w') as f:
            json.dump(fit_data, f, indent=4)

    def predict_runtime(self, n_variables, n_samples):
        """Predict DirectLiNGAM runtime using the accurate complexity formula."""
        M = self.estimate_rank_M(n_samples, n_variables)
        
        # Create features DataFrame with consistent names
        features = pd.DataFrame({
            self.feature_names[0]: [n_samples * (n_variables**3) * (M**2)],
            self.feature_names[1]: [n_variables**4 * M**3]
        })
        
        return max(self.cpu_model.predict(features)[0], 0.075)

    def explain_complexity(self, n_variables, n_samples):
        """Explain the complexity breakdown for given dimensions."""
        M = self.estimate_rank_M(n_samples, n_variables)
        term1 = n_samples * (n_variables**3) * (M**2)
        term2 = n_variables**4 * M**3
        
        print(f"\nComplexity Analysis for {n_variables} variables and {n_samples} samples:")
        print(f"Estimated rank M: {M}")
        print(f"Data-dependent term (np³M²): {term1:.2e}")
        print(f"Structural term (p⁴M³): {term2:.2e}")
        print(f"Ratio (structural/data): {term2/term1:.2f}")
        
        # Predict both CPU and GPU times
        cpu_time = self.predict_runtime(n_variables, n_samples)
        print(f"Predicted CPU time: {cpu_time:.3f} seconds")

if __name__ == "__main__":
    predictor = DirectLiNGAM()
    
    # Test cases
    test_cases = [
        (25, 10000, 7.573),
        (100, 5000, 291.030),
        (5, 5000, 0.075),
        (50, 5000, 36.474),
    ]
    
    print("\nValidation against empirical data:")
    print("vars\tsamples\tdevice\tactual\tpredicted\trel_error")
    print("-" * 70)
    
    total_rel_error = 0
    for vars, samples, actual in test_cases:
        predicted = predictor.predict_runtime(vars, samples)
        rel_error = abs(predicted - actual) / actual * 100
        total_rel_error += rel_error
        print(f"{vars}\t{samples}\t{actual:.3f}\t{predicted:.3f}\t{rel_error:.1f}%")
    
    print(f"\nAverage relative error: {total_rel_error/len(test_cases):.1f}%")
    
    # Analyze some interesting cases
    predictor.explain_complexity(50, 5000)
    predictor.explain_complexity(100, 5000)
    predictor.explain_complexity(200, 10000)