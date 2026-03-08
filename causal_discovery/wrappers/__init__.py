try:
    import torch
except (ImportError, FileNotFoundError):
    torch = None

# All imports are wrapped in try/except so missing optional dependencies
# don't prevent importing individual wrappers that DO have their deps.

# constraint-based algorithms
try:
    from .pc import PC
except (ImportError, FileNotFoundError):
    pass
try:
    from .fci import FCI
except (ImportError, FileNotFoundError):
    pass
try:
    from .cdnod import CDNOD
except (ImportError, FileNotFoundError):
    pass
try:
    from .pc_parallel import PCParallel
except (ImportError, FileNotFoundError):
    pass
try:
    from .inter_iamb import InterIAMB
except (ImportError, FileNotFoundError):
    pass
try:
    from .bamb import BAMB
except (ImportError, FileNotFoundError):
    pass
try:
    from .hiton_mb import HITONMB
except (ImportError, FileNotFoundError):
    pass
try:
    from .iambnpc import IAMBnPC
except (ImportError, FileNotFoundError):
    pass
try:
    from .mbor import MBOR
except (ImportError, FileNotFoundError):
    pass

# score-based algorithms
try:
    from .ges import GES
except (ImportError, FileNotFoundError):
    pass
try:
    from .fges import FGES
except (ImportError, FileNotFoundError):
    pass
try:
    from .x_ges import XGES as XGES
except (ImportError, FileNotFoundError):
    pass
try:
    from .notears_linear import NOTEARSLinear
except (ImportError, FileNotFoundError):
    pass
try:
    from .notears_nolinear import NOTEARSNonlinear
except (ImportError, FileNotFoundError):
    pass
try:
    from .corl import CORL
except (ImportError, FileNotFoundError):
    pass
try:
    from .golem import GOLEM
except (ImportError, FileNotFoundError):
    pass
try:
    from .grasp import GRaSP
except (ImportError, FileNotFoundError):
    pass

# functional-model based algorithms
try:
    from .direct_lingam import DirectLiNGAM
except (ImportError, FileNotFoundError):
    pass
try:
    from .ica_lingam import ICALiNGAM
except (ImportError, FileNotFoundError):
    pass

# hybrid algorithms
try:
    from .hybrid import Hybrid
except (ImportError, FileNotFoundError):
    pass

# time-series algorithms
try:
    from .dynotears import DYNOTEARS
except (ImportError, FileNotFoundError):
    pass
try:
    from .pcmci import PCMCI
except (ImportError, FileNotFoundError):
    pass
try:
    from .var_lingam import VARLiNGAM
except (ImportError, FileNotFoundError):
    pass
try:
    from .granger_causality import GrangerCausality
except (ImportError, FileNotFoundError):
    pass
try:
    from .nts_notears import NTSNOTEARS
except (ImportError, FileNotFoundError):
    pass

constraint_based_algorithms = ['PC', 'FCI', 'CDNOD', 'InterIAMB', 'BAMB', 'HITONMB', 'IAMBnPC', 'MBOR', 'PCParallel', 'AcceleratedPC']
score_based_algorithms = ['GES', 'FGES', 'XGES', 'NOTEARSLinear', 'NOTEARSNonlinear', 'CORL', 'CALM', 'GOLEM', 'DYNOTEARS']
functional_model_based_algorithms = ['DirectLiNGAM', 'ICALiNGAM']

permutation_based_algorithms = ['GRaSP']
hybrid_algorithms = ['Hybrid']
ts_algorithms = ['PCMCI', 'VARLiNGAM', 'DYNOTEARS', 'GrangerCausality', 'NTSNOTEARS']

_ALL_ALGORITHM_NAMES = (constraint_based_algorithms + score_based_algorithms
                        + functional_model_based_algorithms + permutation_based_algorithms
                        + hybrid_algorithms + ts_algorithms)

# Only export names that were actually imported successfully
__all__ = [name for name in _ALL_ALGORITHM_NAMES if name in dir()]
