import os
import random
import shutil


def sample(src_dir, dst_dir, proportion=0.1):
    # list all files in src_dir and subdirectories that ends with .nii.gz
    all_files = []
    os.makedirs(dst_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                all_files.append(os.path.join(root, file))
    # randomly sample a proportion of files
    if len(all_files) == 0:
        print("No .nii.gz files found.")
        return
    sampled_files = random.sample(all_files, min(int(len(all_files) * proportion), len(all_files)))
    # move sampled files to dst_dir
    for file in sampled_files:
        shutil.move(file, os.path.join(dst_dir, os.path.basename(file)))

if __name__ == '__main__':
    sample('/tanghaomiao/medai/data/train/ZR', '/tanghaomiao/medai/data/val/ZR', 0.1)
    sample('/tanghaomiao/medai/data/train/JD', '/tanghaomiao/medai/data/val/JD', 0.1)
    sample('/tanghaomiao/medai/data/train/GY', '/tanghaomiao/medai/data/val/GY', 0.1)
