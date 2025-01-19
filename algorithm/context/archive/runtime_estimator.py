import json
import numpy as np

class RuntimeEstimator:
    def __init__(self):
        self.algo2time_cost = json.load(open('algorithm/context/algo2time_cost.json', encoding='utf-8'))

    def get_time_estimates(self, algo_cans, n_sample, variable):
        """Get runtime estimates for a list of algorithms"""
        estimates = {}
        cdnod_estimates = {}
        
        for algo_can in algo_cans:
            try:
                time_cost = self.time_estimate(algo_can, n_sample, variable)
                estimates[algo_can] = time_cost
                
                if algo_can == 'CDNOD':
                    cdnod_estimates['fisherz'] = time_cost
                    cdnod_estimates['kci'] = self.time_estimate('CDNOD-kci', n_sample, variable)
            except:
                estimates[algo_can] = 'Unknown Time'
                print(f"Meeting Error for {algo_can}")
                
        return estimates, cdnod_estimates
    

    def runtime_estimate(self, algo_can, n_sample, variable):
        

    def format_time_strings(self, estimates, cdnod_estimates):
        """Format time estimates into strings"""
        prompt = "\n".join(f"{algo}: {time}min" for algo, time in estimates.items())
        
        cdnod_prompt = ""
        if cdnod_estimates:
            cdnod_prompt = (
                f"CDNOD using fisherz for indep_test: {cdnod_estimates['fisherz']}min\n"
                f"CDNOD using kci for indep_test: {cdnod_estimates['kci']}min"
            )
            
        return prompt, cdnod_prompt
