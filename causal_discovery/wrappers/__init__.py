try:
    import torch
except ImportError:
    torch = None

# All imports are wrapped in try/except so missing optional dependencies
# don't prevent importing individual wrappers that DO have their deps.

# constraint-based algorithms
try:
    from .pc import PC
except ImportError:
    pass
try:
    from .fci import FCI
except ImportError:
    pass
try:
    from .cdnod import CDNOD
except ImportError:
    pass
try:
    from .pc_parallel import PCParallel
except ImportError:
    pass
try:
    from .inter_iamb import InterIAMB
except ImportError:
    pass
try:
    from .bamb import BAMB
except ImportError:
    pass
try:
    from .hiton_mb import HITONMB
except ImportError:
    pass
try:
    from .iambnpc import IAMBnPC
except ImportError:
    pass
try:
    from .mbor import MBOR
except ImportError:
    pass

# score-based algorithms
try:
    from .ges import GES
except ImportError:
    pass
try:
    from .fges import FGES
except ImportError:
    pass
try:
    from .x_ges import XGES as XGES
except ImportError:
    pass
try:
    from .notears_linear import NOTEARSLinear
except ImportError:
    pass
try:
    from .notears_nolinear import NOTEARSNonlinear
except ImportError:
    pass
try:
    from .corl import CORL
except ImportError:
    pass
try:
    from .golem import GOLEM
except ImportError:
    pass
try:
    from .grasp import GRaSP
except ImportError:
    pass

# functional-model based algorithms
try:
    from .direct_lingam import DirectLiNGAM
except ImportError:
    pass
try:
    from .ica_lingam import ICALiNGAM
except ImportError:
    pass

# hybrid algorithms
try:
    from .hybrid import Hybrid
except ImportError:
    pass

# time-series algorithms
try:
    from .dynotears import DYNOTEARS
except ImportError:
    pass
try:
    from .pcmci import PCMCI
except ImportError:
    pass
try:
    from .var_lingam import VARLiNGAM
except ImportError:
    pass
try:
    from .granger_causality import GrangerCausality
except ImportError:
    pass
try:
    from .nts_notears import NTSNOTEARS
except ImportError:
    pass

constraint_based_algorithms = ['PC', 'FCI', 'CDNOD', 'InterIAMB', 'BAMB', 'HITONMB', 'IAMBnPC', 'MBOR', 'PCParallel', 'AcceleratedPC']
score_based_algorithms = ['GES', 'FGES', 'XGES', 'NOTEARSLinear', 'NOTEARSNonlinear', 'CORL', 'CALM', 'GOLEM', 'DYNOTEARS']
functional_model_based_algorithms = ['DirectLiNGAM', 'ICALiNGAM']

permutation_based_algorithms = ['GRaSP']
hybrid_algorithms = ['Hybrid']
ts_algorithms = ['PCMCI', 'VARLiNGAM', 'DYNOTEARS', 'GrangerCausality', 'NTSNOTEARS']

__all__ = constraint_based_algorithms + score_based_algorithms + functional_model_based_algorithms + permutation_based_algorithms + hybrid_algorithms + ts_algorithms
