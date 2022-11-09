import os
from os.path import basename, dirname, join
from glob import glob

import numpy as np

def get_ct_mean_std_from_traindata(train_subjects, ):
    voxels_all = []
    
    for subject in train_subjects:
        if os.path.exists(subject):
            npz_file = np.load(subject)
            
            ct_brain = npz_file["data"]
            nonzero_mask = npz_file["nonzero_mask"]
            
            voxels_all.append(ct_brain[nonzero_mask][::10])
    
    voxels_all = np.concatenate(voxels_all)
    mean = np.mean(voxels_all)
    std = np.std(voxels_all)
    
    percentile_99_5 = np.percentile(voxels_all, 99.5)
    percentile_00_5 = np.percentile(voxels_all, 00.5)
    
    return mean, std, percentile_99_5, percentile_00_5