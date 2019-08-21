import warnings

from .kmeans import MutualInfoKMeans
from .neural_estimation import MutualInfoNeuralEstimation
from .npeet import MutualInfoNPEET
from .gcmi import MutualInfoGCMI

try:
    from .idtxl_jidt import MutualInfoIDTxl
except ImportError:
    warnings.warn("To use IDTxl, please run 'pip install -r requirements-extra.txt'")
