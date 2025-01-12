import os
from tqdm import tqdm
import numpy as np
import nibabel as nib
from PIL import Image
import argparse
from utils import get_nii_files, get_image_set, get_background_proportion, get_number_of_slices
from collections import Counter


def random_walk(data_dir, target_dir, num_samples=10, slice_idx=[50]):
    # read all files in data_dir (including subdirectories) that ends with .nii.gz
    print(f'Reading files from {data_dir}')
    all_files = get_nii_files(data_dir)

    completed_samples = 0
    while completed_samples < num_samples:
        # randomly sample num_samples files from all_files
        sampled_files = np.random.choice(all_files, size=num_samples-completed_samples, replace=False)
        # create target_dir if not exists
        os.makedirs(target_dir, exist_ok=True)
        for file in tqdm(sampled_files, desc='Extracting slices'):
            try:
                image_set = get_image_set(file, slice_idx)
                number_of_slices = get_number_of_slices(file)
                for image, si in zip(image_set, slice_idx):
                    background_proportion = get_background_proportion(image)
                    image.save(os.path.join(target_dir, f'{si}of{number_of_slices}_{background_proportion:.2f}_' + os.path.basename(file)) + '.png')
                completed_samples += 1
            except IndexError as e:
                print(f'IndexError in file {file}, skipping...')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Randomly sample and extract 2D slices from .nii.gz files')
    parser.add_argument('--data_dir', type=str, default='/tanghaomiao/medai/data',
                        help='Directory containing .nii.gz files')
    parser.add_argument('--target_dir', type=str, default='/tanghaomiao/medai/random_walk',
                        help='Directory to save extracted slices')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of files to sample')
    parser.add_argument('--slice_idx', type=int, nargs='+', default=[50],
                        help='Indices of slices to extract')
    args = parser.parse_args()

    random_walk(args.data_dir, args.target_dir, args.num_samples, args.slice_idx)
