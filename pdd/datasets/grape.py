"""MNIST handwritten digits dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils.data_utils import get_file
import numpy as np


def load_data(path='grape.tar'):
    """Loads the PDD grape dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.pdd/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path,
                    origin="http://pdd.jinr.ru/images/base/grape/grape.tar",
                    file_hash='cbc74aef6e5f20ef22b41d1f0f77e40567e6fd1f4e2c007341df9fb4672c3fd8')

    