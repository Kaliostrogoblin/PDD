"""MNIST handwritten digits dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils.data_utils import get_file
from ..utils.data_utils import datadir_train_test_split
import numpy as np
import sys


def load_data(path='grape.tar', 
              split_on_train_test=False, 
              test_size=None, 
              random_state=0):
    """Loads the PDD grape dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.pdd/datasets).
        split_on_train_test: flag, controls whether or not
            data should be splitted on train and test
        test_size: the size of test data fraction
    # Returns
        Path to the folder with data or tuple with train and test paths
    """
    path = get_file(path,
                    origin="http://pdd.jinr.ru/images/base/grape/grape.tar",
                    file_hash='3127fa909fbbff8dc3b7e21c3901faec389e1d8297018050b023b7b46b1328f7',
                    extract=True)

    try:
        if split_on_train_test:
            print("Splitting on train and test...")
            test_size = 0.15 if test_size is None else test_size
            train_path, test_path = datadir_train_test_split(
                path, test_size, random_state)
            return (train_path, test_path)
    except:
        print("Unexpected error:", sys.exc_info()[0])
    
    return path


