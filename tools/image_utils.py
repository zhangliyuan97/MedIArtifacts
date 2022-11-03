from PIL import Image
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import matplotlib


def concat_images(image_files, save_dir):
    # image_files: PIL object list
    COL = 4
    ROW = 50
    UNIT_HEIGHT_SIZE = 450
    UNIT_WIDTH_SIZE = 450
    SAVE_QUALITY = 50

    target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW))
    for row in range(ROW):
        for col in range(COL):
            target.paste(image_files[COL * row + col], [0 + UNIT_WIDTH_SIZE * col, 0 + UNIT_HEIGHT_SIZE * row])
    target.save(join(save_dir, "result.png"), quality=SAVE_QUALITY)


def normalize_intensity(img):
    img = img / (img.max() - img.min + 1e-10)
    return img * 255.0


def plot_seg_result(b1_file, seg_file, fig_save_path=None):
    b1_data = nib.load(b1_file).get_fdata()
    seg_data = nib.load(seg_file).get_fdata()
    print(b1_data.shape, seg_data.shape)
    slice_number = b1_data.shape[-1]
    print(slice_number)

    # process b1 data
    b1_data = np.transpose(b1_data, (1, 0, 2))
    b1_data = np.flip(b1_data, 0)

    # process seg data
    seg_data = np.transpose(seg_data, (1, 0, 2))
    seg_data = np.flip(seg_data, 0)

    matplotlib.use('AGG')
    fig = plt.figure(figsize=(3, slice_number))

    j = 1
    for i in range(1, slice_number):
        # plot b1 image
        ax = fig.add_subplot(slice_number, 3, j)
        plt.axis("off")
        plt.imshow(np.float32(b1_data[:, :, i - 1]), cmap="gray")
        j += 1

        # plot seg image
        ax = fig.add_subplot(slice_number, 3, j)
        plt.axis("off")
        plt.imshow(np.float32(seg_data[:, :, i - 1]), cmap="gray")
        j += 1

        # plot overlap seg image and b1 image
        ax = fig.add_subplot(slice_number, 3, j)
        plt.axis("off")
        plt.imshow(np.float32(b1_data[:, :, i - 1]), cmap="gray")
        plt.imshow(np.float32(seg_data[:, :, i - 1]), cmap="Reds", alpha=0.8)
        j += 1

    plt.subplots_adjust(wspace=0.0, hspace=0.2)
    plt.savefig(fig_save_path, bbox_inche="tight", dpi=600)
    plt.close(fig)
