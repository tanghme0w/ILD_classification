import os
from tqdm import tqdm
import numpy as np
import nibabel as nib
from PIL import Image
from collections import Counter

def get_image_set(file_path, slice_idx: list[int] = None):
    # load the data from file_path
    data = nib.load(file_path).get_fdata()
    # get the number of slices
    num_slices = data.shape[-1]
    # return a list of images
    if slice_idx is None:
        slice_idx = range(num_slices)
    return [get_image_slice(data, i) for i in slice_idx]


def get_image_slice(data, slice_idx):
    # get the slice at slice_idx
    slice = data[:, :, slice_idx]
    # normalize the slice to [0, 255]
    if np.max(slice) - np.min(slice) == 0:
        slice = np.zeros_like(slice)
    else:
        slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice)) * 255
    # return the slice as an image
    return Image.fromarray(slice).convert('RGB')


def get_number_of_slices(file_path):
    # load the data from file_path
    data = nib.load(file_path).get_fdata()
    # get the number of slices
    return data.shape[2]


def get_background_proportion(image: Image.Image):
    # find the most frequent color in the slice
    color_counts = Counter(image.getdata())
    most_common_color = color_counts.most_common(1)[0]
    return most_common_color[1] / len(image.getdata())


def get_nii_files(data_dir):
    # read all files in data_dir (including subdirectories) that ends with .nii.gz
    all_files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in tqdm(filenames, desc='Reading files'):
            if filename.endswith('.nii.gz'):
                all_files.append(os.path.join(root, filename))
    return all_files