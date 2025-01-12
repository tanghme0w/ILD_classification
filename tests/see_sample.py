import os
from utils import get_image_set, get_number_of_slices

SAMPLE_PATH = '/tanghaomiao/medai/data/train/ZR/202208221215.nii.gz'

slices_count = get_number_of_slices(SAMPLE_PATH)
image_set = get_image_set(SAMPLE_PATH, slice_idx=range(slices_count))

os.makedirs('202209021399_images', exist_ok=True)
for i, image in enumerate(image_set):
    image.save(f'202209021399_images/test_{i}.png')