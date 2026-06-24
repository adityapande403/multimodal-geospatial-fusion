"""Multimodal Geospatial Data Fusion Package"""

__version__ = "1.0.0"
__author__ = "Geospatial AI/ML Research Team"

from . import preprocessing
from . import fusion
from . import change_detection
from . import evaluation
from . import utils

__all__ = [
    'preprocessing',
    'fusion',
    'change_detection',
    'evaluation',
    'utils'
]
