from os.path import basename, dirname, join
from glob import glob

import numpy as np


dataset_npz_path = "data/hematoma_npz"

subjects = list( glob( join(dataset_npz_path, "*.npz") ) )