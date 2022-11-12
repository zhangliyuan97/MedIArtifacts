import os
from os.path import basename, dirname, join

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from glob import glob


class hematoma_loader(Dataset):
    def __init__(self, sub_root, csv_dataframe, img_size, is_transform, transforms=None, use_seg=False, is_train=False):
        self.sub_root = sub_root
        self.img_size = img_size
        self.is_transform = is_transform
        self.transforms = transforms
        self.use_seg = use_seg
        self.is_train = is_train   # this will control pad_or_crop & transforms whether or not
        
        # for ct images, we just statistic the foreground pixels to find the mean and std previously
        # self.mean = mean
        # self.std = std
        # self.clip_high = clip_high
        # self.clip_low = clip_low
        
        # select train/valid/test subjects from .csv file
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
    
    def _normalize_over_given_clip(self, img):
        img = np.clip(img, 0, 100)
        
        img = (img / 100) * 2.0 - 1.0
        
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
                slice_start = np.random.randint(0, image.shape[i] - img_size[i] + 1)
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
        
        img_ct = self._normalize_over_given_clip(img_ct)
        
        if self.is_train:
            inputs = self._get_train_valid_item(img_ct)
        else:
            inputs = self._get_inference_item(img_ct)  # inputs_multi_patches
            
        return {
            "inputs": inputs,
            "cls_labels": cls_label,
            "file_path": self.npz_files[index], 
        }
    
    def _get_train_valid_item(self, normalized_img):
        """For training/validation time, we only crop each image to self.img_size, and augment the cropped image only once
        """
        img_ct = self._pad_or_crop_to_img_size(normalized_img, seg=None, img_size=self.img_size, mode="constant")["padded_img"]
            
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
        
        return inputs
    
    def _get_inference_item(self, normalized_img):
        """For inference time, batch size will be set to only 1.
        And we need to crop the whole image to multiple overlapped patches
        """
        c, h, w = normalized_img.shape
        target_c, target_h, target_w = self.img_size
        c_stride = c - target_c
        h_stride = h - target_h
        w_stride = w - target_w
        
        # firstly, pad image if image.shape smaller than required self.img_size
        to_padding = [[0, 0] for dim in range(3)]
        if c_stride <= 0:
            to_padding[0][0] = abs(c_stride) // 2
            to_padding[0][1] = abs(c_stride) - to_padding[0][0]
        if h_stride <= 0:
            to_padding[1][0] = abs(h_stride) // 2
            to_padding[1][1] = abs(h_stride) - to_padding[1][0]
        if w_stride <= 0:
            to_padding[2][0] = abs(w_stride) // 2
            to_padding[2][1] = abs(w_stride) - to_padding[2][0]
        padded_img = np.pad(normalized_img, to_padding, mode="constant", constant_values=0)
        
        # secondly, crop the image to multiple patches, here image.shape >= self.img_size
        padded_c, padded_h, padded_w = padded_img.shape
        slice_start_set_c, slice_start_set_h, slice_start_set_w = set([0]), set([0]), set([0])
        slice_start_set_c.add(padded_c - target_c)
        slice_start_set_h.add(padded_h - target_h)
        slice_start_set_w.add(padded_w - target_w)
        
        patches = []
        for slice_start_c in slice_start_set_c:
            for slice_start_h in slice_start_set_h:
                for slice_start_w in slice_start_set_w:
                    patch_slice = [
                        slice(slice_start_c, slice_start_c + target_c), 
                        slice(slice_start_h, slice_start_h + target_h),
                        slice(slice_start_w, slice_start_w + target_w),
                    ]
                    image_patch = self._to_tensor(padded_img[tuple(patch_slice)])
                    # seg = self._to_tensor(np.squeeze(seg)) - 0.5
        
                    if self.use_seg:
                        # inputs = torch.cat([img_ct, seg])
                        pass
                    else:
                        inputs = image_patch
                        
                    patches.append(inputs)
        
        return patches
        
        
    def __len__(self):
        return len(self.npz_files)