import numpy as np
from PIL import Image


def scale_intensity(arr):
    arr = arr / (arr.max() - arr.min() + 1e-10)
    return arr * 255.0


def visual_arr(img_arr, save_filename):
    img_pil = Image.fromarray(img_arr)
    img_pil = Image.fromarray(img_arr).convert('RGB')
    img_pil.save(save_filename, quality=50)


def visual_npz(npz_path):
    npz_data = np.load(npz_path)
    dwi_arr = npz_data['dwi'][10,...]
    flair_arr = npz_data['flair'][10, ...]
    visual_arr(dwi_arr, "./dwi.png")
    visual_arr(flair_arr, "./flair.png")


if __name__ == '__main__':
    visual_npz("")
