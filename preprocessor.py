import os
import glob
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from collections import OrderedDict
from brain_extractor import BrainExtractor
from scipy.ndimage.interpolation import map_coordinates
from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize


class Preprocessor(object):
    def __init__(self, target_spacing):
        self.target_spacing = target_spacing

    def array2image(self, array, origin_image, new_spacing=None):
        rec_image = sitk.GetImageFromArray(array)
        rec_image.SetDirection(origin_image.GetDirection())
        if new_spacing is not None:
            rec_image.SetSpacing(new_spacing)
        else:
            rec_image.SetSpacing(origin_image.GetSpacing())
        rec_image.SetOrigin(origin_image.GetOrigin())

        return rec_image

    def resample_data_or_seg(self, data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, order_z=0):
        """
        separate_z=True will resample with order 0 along z
        :param data:
        :param new_shape:
        :param is_seg:
        :param axis:
        :param order:
        :param do_separate_z:
        :param order_z: only applies if do_separate_z is True
        :return:
        """
        assert len(data.shape) == 4, "data must be (c, x, y, z)"
        assert len(new_shape) == len(data.shape) - 1
        if is_seg:
            resize_fn = resize_segmentation
            kwargs = OrderedDict()
        else:
            resize_fn = resize
            kwargs = {'mode': 'edge', 'anti_aliasing': False}
        dtype_data = data.dtype
        shape = np.array(data[0].shape)
        new_shape = np.array(new_shape)
        if np.any(shape != new_shape):
            data = data.astype(float)
            if do_separate_z:
                print("separate z, order in z is", order_z, "order inplane is", order)
                assert len(axis) == 1, "only one anisotropic axis supported"
                axis = axis[0]
                if axis == 0:
                    new_shape_2d = new_shape[1:]
                elif axis == 1:
                    new_shape_2d = new_shape[[0, 2]]
                else:
                    new_shape_2d = new_shape[:-1]

                reshaped_final_data = []
                for c in range(data.shape[0]):
                    reshaped_data = []
                    for slice_id in range(shape[axis]):
                        if axis == 0:
                            reshaped_data.append(
                                resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                        elif axis == 1:
                            reshaped_data.append(
                                resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                        else:
                            reshaped_data.append(
                                resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs).astype(dtype_data))
                    reshaped_data = np.stack(reshaped_data, axis)
                    if shape[axis] != new_shape[axis]:

                        # The following few lines are blatantly copied and modified from sklearn's resize()
                        rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                        orig_rows, orig_cols, orig_dim = reshaped_data.shape

                        row_scale = float(orig_rows) / rows
                        col_scale = float(orig_cols) / cols
                        dim_scale = float(orig_dim) / dim

                        map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                        map_rows = row_scale * (map_rows + 0.5) - 0.5
                        map_cols = col_scale * (map_cols + 0.5) - 0.5
                        map_dims = dim_scale * (map_dims + 0.5) - 0.5

                        coord_map = np.array([map_rows, map_cols, map_dims])
                        if not is_seg or order_z == 0:
                            reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                       mode='nearest')[None].astype(dtype_data))
                        else:
                            unique_labels = np.unique(reshaped_data)
                            reshaped = np.zeros(new_shape, dtype=dtype_data)

                            for i, cl in enumerate(unique_labels):
                                reshaped_multihot = np.round(
                                    map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                    mode='nearest'))
                                reshaped[reshaped_multihot > 0.5] = cl
                            reshaped_final_data.append(reshaped[None].astype(dtype_data))
                    else:
                        reshaped_final_data.append(reshaped_data[None].astype(dtype_data))
                reshaped_final_data = np.vstack(reshaped_final_data)
            else:
                print("no separate z, order", order)
                reshaped = []
                for c in range(data.shape[0]):
                    reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None].astype(dtype_data))
                reshaped_final_data = np.vstack(reshaped)
            return reshaped_final_data.astype(dtype_data)
        else:
            print("no resampling necessary")
            return

    def get_image_slicer_to_crop(self, nonzero_mask):
        outside_value = 0
        mask_voxel_coords = np.where(nonzero_mask != outside_value)
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
        return resizer

    def get_new_shape(self, itk_img, cropped_img, target_spacing):
        orig_spacing = itk_img.GetSpacing()
        cropped_shape = cropped_img.shape
        ratio_orig_target = (np.array(orig_spacing) / np.array(target_spacing)).astype(np.float)
        calibrate_ratio_order = [ratio_orig_target[2], ratio_orig_target[0], ratio_orig_target[1]]
        new_shape = np.round((np.array(calibrate_ratio_order) * cropped_shape)).astype(int)
        return new_shape

    def crop_image(self, itk_img):
        img = sitk.GetArrayFromImage(itk_img).astype(np.float)
        threshold = np.percentile(img, 50)
        brain_extractor = BrainExtractor()
        nonzero_mask = brain_extractor.get_brain_mask(itk_img, th=threshold)
        nonzero_mask = sitk.GetArrayFromImage(nonzero_mask)
        resizer = self.get_image_slicer_to_crop(nonzero_mask)
        cropped_img = img[resizer]
        cropped_nonzero_mask = nonzero_mask[resizer]
        return cropped_img, cropped_nonzero_mask

    def resample_patient(self, itk_img, cropped_img):
        new_shape = self.get_new_shape(itk_img, cropped_img, self.target_spacing)
        cropped_img = np.expand_dims(cropped_img, axis=0)
        resampled_img = self.resample_data_or_seg(cropped_img, new_shape, is_seg=False, axis=[0],
                                                  do_separate_z=True, order_z=0)
        return self.array2image(resampled_img, itk_img, self.target_spacing)

    def run(self, img_path):
        itk_img = sitk.ReadImage(img_path)
        cropped_img, cropped_nonzero_mask = self.crop_image(itk_img)
        resampled_img = self.resample_patient(itk_img, cropped_img)
        return resampled_img
