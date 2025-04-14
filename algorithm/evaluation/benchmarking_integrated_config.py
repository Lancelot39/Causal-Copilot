from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class BenchmarkConfig:
    # Data directory
    data_dir: str = "simulated_data/copilot_benchmark_v4" # /20250207_030932_uniform_noise_seed_0_nodes25_samples5000"
    # Output directory
    output_dir: str = "simulated_data/copilot_benchmark_v4_results"
    
    # # Default experiment settings
    # n_vars: int = 25
    # n_samples: int = 5000
    # edge_prob: float = 0.2
    # functional_form: str = "linear"
    # noise_type: str = "gaussian"
    # noise_scale: float = 0.1
    # heterogeneity: Optional[int] = None
    # discrete_ratio: float = 0.0
    # measurement_error_ratio: float = 0.0
    # measurement_error_scale: float = 0.1
    # missing_ratio: float = 0.0

    # # Scale comparison settings
    # variable_sizes: List[int] = field(default_factory=lambda: [5, 10, 25, 50, 100])
    # sample_sizes: List[int] = field(default_factory=lambda: [1000, 2500, 5000, 10000])
    
    # # Graph density settings
    # edge_probabilities: List[float] = field(default_factory=lambda: [10, 20, 30])
    
    # # Functional type settings
    # functional_types: List[str] = field(default_factory=lambda: ["linear", "nonlinear"])
    
    # # Noise type settings
    # noise_types: List[str] = field(default_factory=lambda: ["gaussian", "uniform"])
    
    # # Heterogeneity settings
    # domain_counts: List[Optional[int]] = field(default_factory=lambda: [None, 5, 10, 15])
    
    # # Data quality settings
    # discrete_ratios: List[float] = field(default_factory=lambda: [0.1, 0.2])
    # measurement_error_ratios: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    # missing_ratios: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])

    # # Algorithm settings
    # algorithms: List[str] = field(default_factory=lambda: ['PC', 'FCI', 'CDNOD', 'AcceleratedPC', 
    #                                                        'GES', 'DirectLiNGAM', 'AcceleratedDirectLiNGAM', 
    #                                                        'GOLEM', 'CORL', 
    #                                                     #    'Hybrid', 'CALM', 'AcceleratedCDNOD', 'ICALiNGAM',
    #                                                        'NOTEARSLinear', 'NOTEARSNonlinear',
    #                                                        'FGES', 'XGES', 'GRaSP', 'HITONMB', 'BAMB', 'IAMBnPC', 'MBOR', 'InterIAMB'])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary format"""
        return {
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            # "default": {
            #     "n_vars": self.n_vars,
            #     "n_samples": self.n_samples,
            #     "edge_prob": self.edge_prob,
            #     "functional_form": self.functional_form,
            #     "noise_type": self.noise_type,
            #     "noise_scale": self.noise_scale,
            #     "heterogeneity": self.heterogeneity,
            #     "discrete_ratio": self.discrete_ratio,
            #     "measurement_error_ratio": self.measurement_error_ratio,
            #     "measurement_error_scale": self.measurement_error_scale,
            #     "missing_ratio": self.missing_ratio
            # },
            # "scale_comparison": {
            #     "variable_sizes": self.variable_sizes,
            #     "sample_sizes": self.sample_sizes
            # },
            # "graph_density": {
            #     "edge_probabilities": self.edge_probabilities
            # },
            # "functional_types": self.functional_types,
            # "noise_types": self.noise_types,
            # "heterogeneity": {
            #     "domain_counts": self.domain_counts
            # },
            # "data_quality": {
            #     "discrete_ratios": self.discrete_ratios,
            #     "measurement_error_ratios": self.measurement_error_ratios,
            #     "missing_ratios": self.missing_ratios
            # },
            # "algorithms": self.algorithms
        }

def get_config() -> Dict[str, Any]:
    """
    Creates and returns a BenchmarkConfig instance converted to dictionary format.
    This maintains backward compatibility with existing code.
    """
    config = BenchmarkConfig()
    return config.to_dict() 