# MedIArtifacts

This is a repo that stores the useful codes for medical image preprocessing.

# How to use?

## Preprocessing

* The mainly important function is preprocessor.run(), and you can do preprocess simply by giving a image_path to this
  function.
* 'multiprocess_preprocess.py' is a demo for preprocesss DWI and Flair data using multiprocess.
* The preprocess pipeline mainly include calculate brain mask, intensity normalization, crop ROI and resample.

## cls_task
This project implement the framework for classfication task.
