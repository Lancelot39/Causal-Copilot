
from fges import FGES
from SEMScore import SEMBicScore
from search_util import *
import numpy as np

class SemEstimator:

    def __init__(self, dataset, sparsity=2, savename=None):
        ''' Initialize a SEM Estimator '''
        self.dataset = dataset
        self.sparsity = sparsity
        self.savename = savename

        self.pattern = None
        self.dag = None
        self.penalty = None
        self.params = None
        self.residuals = None
        self.graph_cov = None
        self.true_cov = np.cov(dataset.transpose())

    def set_pattern(self, pattern):
        self.pattern = pattern
        self.dag = None

    def search(self, verbose=False, cache_interval=1):
        ''' Run an FGES search '''
        score = SEMBicScore(self.sparsity, dataset=self.dataset, cache_interval=cache_interval)
        self._fges = FGES(list(range(self.dataset.shape[1])), score, save_name=self.savename, verbose=verbose)
        self._fges.search()
        self.set_pattern(self._fges.graph)

    def get_dag(self):
        if self.dag is None:
            if self.pattern is None:
                self.search()
            self.dag, self.penalty = dagFromPatternWithColliders(self.pattern)
        return self.dag

    def estimate(self):
        ''' Estimate edge weights '''
        self.get_dag()

        self.params, self.residuals = estimate_parameters(self.dag, self.dataset)
        self.graph_cov = get_covariance_matrix(self.params, self.residuals)
