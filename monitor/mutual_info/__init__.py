from .kmeans import MutualInfoKMeans
from .neural_estimation import MutualInfoNeuralEstimation
from .npeet import MutualInfoNPEET

try:
    from .idtxl_jidt import MutualInfoIDTxl
except ImportError:
    print("To use IDTxl, please run 'pip install -r requirements-extra.txt'")
    pass
