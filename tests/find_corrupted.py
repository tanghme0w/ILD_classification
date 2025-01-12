import os
import nibabel as nib
from tqdm import tqdm
from utils import get_nii_files


def find_corrupted_data(data_dir, min_dim=50):
    """Find possibly corrupted .nii.gz files in the given directory.

    Returns:
        list: A list of paths to possibly corrupted files.
    """
    corrupted_files = []
    for file in tqdm(get_nii_files(data_dir), desc='Finding corrupted files'):
        data = nib.load(file).get_fdata()
        if data.shape[0] < min_dim or data.shape[1] < min_dim:
            corrupted_files.append(file)
    return corrupted_files


if __name__ == '__main__':
    data_dir = '/tanghaomiao/medai/data'
    corrupted_files = find_corrupted_data(data_dir, min_dim=50)
    print(f"Found {len(corrupted_files)} corrupted files")
    for file in corrupted_files:
        print(file)
