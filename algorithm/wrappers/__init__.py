import torch
from .pc import PC
from .fci import FCI
from .cdnod import CDNOD
from .ges import GES
from .fges import FGES
from .xges import XGES
from .direct_lingam import DirectLiNGAM
from .ica_lingam import ICALiNGAM
from .notears import NOTEARS

if torch.cuda.is_available():
    # Import AcceleratedLiNGAM if GPU is available
    from .accelerated_lingam import AcceleratedDirectLiNGAM
    __all__ = ['PC', 'FCI', 'CDNOD', 'GES', 'FGES', 'XGES', 'DirectLiNGAM', 'AcceleratedDirectLiNGAM', 'ICALiNGAM', 'NOTEARS']
else:
    __all__ = ['PC', 'FCI', 'CDNOD', 'GES', 'FGES', 'XGES', 'DirectLiNGAM', 'ICALiNGAM', 'NOTEARS']