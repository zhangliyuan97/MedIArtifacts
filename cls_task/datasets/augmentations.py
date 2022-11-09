import numpy as np
from batchgenerators.transforms.color_transforms import GammaTransform, ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
from batchgenerators.transforms.abstract_transforms import Compose


def get_transformations(cfg):
    img_size = tuple(cfg['data']['img_size'])
    transforms_string = cfg['data']['transforms']

    transforms = []
    if "gamma" in transforms_string:
        transforms.append(GammaTransform(gamma_range=(0.7, 1.5), retain_stats=True, p_per_sample=.3))

    if "contrast" in transforms_string:
        transforms.append(
            ContrastAugmentationTransform(contrast_range=(.7, 1.25), preserve_range=True, p_per_sample=.3))

    if 'mirror' in transforms_string:
        transforms.append(MirrorTransform(axes=(2,), p_per_sample=.5))

    do_rotation = True if 'rotate' in transforms_string else False
    do_scale = True if 'scale' in transforms_string else False
    do_crop = True if 'crop' in transforms_string else False
    transforms.append(SpatialTransform(
        img_size, patch_center_dist_from_border=(np.array(img_size) // 2),
        
        do_elastic_deform=False, alpha=(0., 900.), sigma=(9.0, 13.0),
        
        do_rotation=do_rotation, angle_x=(-0.5235987755982988, 0.5235987755982988),
        angle_y=(0, 0), angle_z=(0, 0), p_rot_per_axis=1., p_rot_per_sample=.2,
        
        do_scale=do_scale, scale=(.8, 1.2),
        border_mode_data='constant', border_cval_data=0, order_data=3,
        border_mode_seg='constant', border_cval_seg=0, order_seg=1, p_scale_per_sample=.2,
        
        random_crop=do_crop, independent_scale_for_each_axis=False,
    ))

    if 'noise' in transforms_string:
        transforms.append(GaussianNoiseTransform(noise_variance=(0, .1), p_per_sample=.2))
    if 'blur' in transforms_string:
        transforms.append(
            GaussianBlurTransform(blur_sigma=(.5, 1.0), different_sigma_per_channel=True, p_per_channel=.5,
                                  p_per_sample=.2))

    return Compose(transforms)