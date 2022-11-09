import os
from os.path import basename, dirname, join

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from glob import glob


class hematoma_loader(Dataset):
    def __init__(self, sub_root, csv_dataframe, img_size, is_transform, mean, std, clip_high, clip_low, transforms=None, use_seg=False):
        self.sub_root = sub_root
        self.img_size = img_size
        self.is_transform = is_transform
        self.transforms = transforms
        self.use_seg = use_seg
        
        # for ct images, we just statistic the foreground pixels to find the mean and std previously
        self.mean = mean
        self.std = std
        self.clip_high = clip_high
        self.clip_low = clip_low
        
        self.subjects = sorted(list(glob(join(self.sub_root, "*.npz"))))
        dataframe_subject_ids = np.array(csv_dataframe["patient_id"])
        dataframe_labels = np.array(csv_dataframe["label"])
        print("subjects count: ", len(self.subjects), "dataframe shape: ", dataframe_subject_ids.shape)
        
        self.npz_files = []
        self.targets = []
        # we select the overlap between subjects and dataframe
        for subject in self.subjects:
            subject_id = basename(subject).split(".")[0]
            if subject_id in dataframe_subject_ids:
                self.npz_files.append(subject)
                label = dataframe_labels[dataframe_subject_ids == subject_id]
                self.targets.append(label)
        
        a = np.array(self.targets)
        print("> Hematoma label distribution: ", (a == 0).sum(), (a == 1).sum())
    
    def _normalize(self, img, nonzero_mask):
        # clip in this img self
        percentile_99_5 = np.percentile(img, 99.5)
        percentile_00_5 = np.percentile(img, 00.5)
        img = np.clip(img, percentile_00_5, percentile_99_5)
        
        brain_mask = nonzero_mask > 0
        brain_mean = img[brain_mask].mean()
        brain_std = img[brain_mask].std()
        img = (img - brain_mean) / brain_std
        
        return img
    
    def _normalize_over_voxels_all(self, img, ):
        # clip img over voxels all percentiles
        img = np.clip(img, self.clip_low, self.clip_high)
        
        img = (img - self.mean) / self.std
        
        return img
    
    def _pad_or_crop_to_img_size(self, image, seg, img_size, mode="constant"):
        """Image cropping to img_size
        """
        rank = len(img_size)
        
        # Create placeholders for the new shape
        from_indices = [[0, image.shape[dim]] for dim in range(rank)]  # [ [0, 0], [0, 1], [0, 2] ]
        to_padding = [[0, 0] for dim in range(rank)]
        
        slicer = [slice(None)] * rank
        
        # for each dimensions find whether it is supposed to be cropped or padded
        for i in range(rank):
            if image.shape[i] <= img_size[i]:
                to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
                to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
            else:
                slice_start = np.random.randint(0, image.shape[i] - img_size[i])
                from_indices[i][0] = slice_start
                from_indices[i][1] = from_indices[i][0] + img_size[i]
            
            # Create slicer object to crop or leach each dimension
            slicer[i] = slice(from_indices[i][0], from_indices[i][1])
        
        padded_img = np.pad(image[tuple(slicer)], to_padding, mode=mode, constant_values=0)
        
        return {
            "padded_img": padded_img
        }
    
    def _to_tensor(self, img):
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = torch.from_numpy(img)
        
        return img
    
    def __getitem__(self, index: int):
        img_seg_npz = np.load(self.npz_files[index])
        cls_label = self.targets[index]
        
        img_ct = img_seg_npz["data"]
        
        img_ct = self._normalize_over_voxels_all(img_ct)

        img_ct = self._pad_or_crop_to_img_size(img_ct, seg=None, img_size=self.img_size, mode="constant")["padded_img"]
        
        data = img_ct[None, None, ...]
        if self.transforms and self.is_transform:
            data_dict = self.transforms(**{"data": data})
            data = np.squeeze(data_dict["data"])
        
        img_ct = np.squeeze(data)
        img_ct = self._to_tensor(img_ct)
        # seg = self._to_tensor(np.squeeze(seg)) - 0.5
        
        if self.use_seg:
            # inputs = torch.cat([img_ct, seg])
            pass
        else:
            inputs = img_ct
            
        return {
            "inputs": inputs,
            "cls_labels": cls_label,
            "file_path": self.npz_files[index], 
        }
        
    def __len__(self):
        return len(self.npz_files)