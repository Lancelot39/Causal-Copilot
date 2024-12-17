import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple

class CausalDiscoveryAlgorithm:
    def __init__(self, params: Dict):
        self._params = {}
        self._params.update(params)

    def get_params(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def get_primary_params(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def get_secondary_params(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def test_algorithm(self):
        raise NotImplementedError("This method should be implemented by subclasses")

    @property
    def name(self):
        raise NotImplementedError("This property should be implemented by subclasses")