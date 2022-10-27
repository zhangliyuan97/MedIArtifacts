from PIL import Image
from os.path import join


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
