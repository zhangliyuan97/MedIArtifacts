import importlib
from glob import glob
from os.path import basename, dirname, join

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import DataLoader

from .augmentations import get_transformations
from .cls_balanced_sampler import DistributedWeightedSampler, get_balanced_batch_sampler
from .datasets_utils.get_ct_mean_std import get_ct_mean_std_from_traindata


def find_dataset_using_name(name):
    """Import the module "datasets/[dataset_name]_loader.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + name + "_loader"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = name + '_loader'
    for _name, cls in datasetlib.__dict__.items():
        if _name.lower() == target_dataset_name.lower():
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def create_dataset(cfg):
    data_loader = HematomaDataLoader(cfg)

    dataset = data_loader.load_data()

    return dataset


class HematomaDataLoader():
    "Wrapper class of Dataset class that performs multi-threadd data loading"
    
    def __init__(self, cfg):
        """Initialize this class
        
        Step 1: create a dataset instance given the name [data:name]
        Step 2: create a multi-threaded data loader.
        """
        
        self.cfg = cfg
        self.random_seed = self.cfg["random_seed"]
        
        data_config = self.cfg["data"]
        data_path = data_config["npz_dir"]
        csv_file = data_config["csv_file"]
        hospital = data_config["hospital"]
        fold = data_config["fold"]
        is_zq = data_config["is_zq"]
        
        if is_zq:
            df_dict = self.get_zhangqiang_splits(csv_file)
        else:
            df_dict = self.get_fold_df(fold=fold, csv_file=csv_file, hospital=hospital, random_seed=self.random_seed)
        
        # for ct dataset, we normalize the single image based on the overall statistics
        train_subjects = self.get_train_subjects(df_dict["train"], data_path)
        train_mean, train_std, clip_high, clip_low = get_ct_mean_std_from_traindata(train_subjects)
        
        def _init_fn(worker_id):
            np.random.seed(self.random_seed + worker_id)
            
        transforms = get_transformations(self.cfg)
        use_seg = data_config["use_seg"]
        is_transform = data_config["is_transform"]
        hematoma_dataset = find_dataset_using_name(data_config["name"])
        
        self.train_dataset = hematoma_dataset(
            data_path, df_dict["train"], tuple(data_config["img_size"]), is_transform=is_transform,
            transforms=transforms, use_seg=use_seg, is_train=True,
        )
        self.valid_dataset = hematoma_dataset(
            data_path, df_dict["valid"], tuple(data_config["img_size"]), is_transform=False,
            transforms=transforms, use_seg=use_seg, is_train=True,
        )
        self.test_dataset = hematoma_dataset(
            data_path, df_dict["test"], tuple(data_config["img_size"]), is_transform=False,
            transforms=transforms, use_seg=use_seg, is_train=False,
        )

        # construct sampler for class-imbalanced data
        train_sampler = None
        if not self.cfg['data']['is_shuffle_batch']:
            if self.cfg['data']['train_sampler'] == 'hard':  # class-imbalanced data sampled balanced in mini-batch
                print("Notice you are using HARD batch sampler")
                train_sampler = get_balanced_batch_sampler(self.train_dataset)
            else:
                print("The specified sampler {} have not implemented yet!!".format(self.cfg['data']['train_sampler']))
        
        # construct dataloader for train/valid/test
        self.train_loader = DataLoader(
            self.train_dataset,
            self.cfg['data']['batch_size'],
            shuffle=False if train_sampler else True,
            num_workers=self.cfg['data']['num_workers'],
            pin_memory=True, drop_last=True,
            sampler=train_sampler, worker_init_fn=_init_fn
        )

        valid_sampler = None
        self.valid_loader = DataLoader(
            self.valid_dataset,
            1,
            shuffle=False,
            num_workers=self.cfg['data']['num_workers'],
            pin_memory=True, drop_last=False,
            sampler=valid_sampler, worker_init_fn=_init_fn
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            1,
            shuffle=False,
            num_workers=self.cfg['data']['num_workers'],
            pin_memory=True, drop_last=False,
            worker_init_fn=_init_fn
        )


    def get_train_subjects(self, train_dataframe, data_path):
        subjects = sorted(list(glob(join(data_path, "*.npz"))))
        train_patients = np.array(train_dataframe["patient_id"])
        
        train_subjects = []
        # we select the overlap between subjects and dataframe
        for subject in subjects:
            if basename(subject).split(".")[0] in train_patients:
                train_subjects.append(subject)
        
        return train_subjects
    
    def get_fold_df(self, fold, csv_file, hospital, random_seed=42):
        hematoma_df = pd.read_csv(csv_file)
        print("> read csv file from ", csv_file)
        
        # =============================================
        # you should first select data from the hospital
        # then do the rest train/valid/test splits
        # ... ...
        # for coding fastly, we just step over this process
        # =============================================
        
        # train/valid/test split
        train_x, test_x, train_y, test_y = train_test_split(
            hematoma_df["patient_id"], hematoma_df["label"], test_size=0.2, 
            stratify=hematoma_df["label"], random_state=random_seed)
        
        train_x, valid_x, train_y, valid_y = train_test_split(
            train_x, train_y, test_size=0.2, stratify=train_y, random_state=random_seed)
        
        print(train_x.shape, valid_x.shape, test_x.shape)
        
        train_df = pd.concat([train_x, train_y], axis=1)
        valid_df = pd.concat([valid_x, valid_y], axis=1)
        test_df = pd.concat([test_x, test_y], axis=1)
        
        return {
            "train": train_df,
            "valid": valid_df,
            "test": test_df,
        }
    
    def get_zhangqiang_splits(self, csv_file):
        hematoma_df = pd.read_csv(csv_file)
        print("> read csv file from ", csv_file)

        train_df = hematoma_df[hematoma_df["dataset_group"] == 1]
        valid_df = hematoma_df[hematoma_df["dataset_group"] == 2]
        test_df = hematoma_df[hematoma_df["dataset_group"] == 3]
        
        return {
            "train": train_df,
            "valid": valid_df,
            "test": test_df,
        }
    
    def load_data(self):
        return self
        