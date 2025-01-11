import os
import shutil
from tqdm import tqdm


def get_data(filepath):
    """ read csv file and return a dictionary with the number of lines, words, and characters

    Args:
        filepath (str): the path to the csv file

    Returns:
        three lists, each is a column in the csv file
    """

    lines = []
    with open(filepath, 'r') as file:
        content = file.readlines()
    
    for i, line in enumerate(content):
        # ignore first two lines
        if i < 2:
            continue
        entry = line.strip().split(',')
        lines.append(entry)

    # split the lines into three lists  
    centers = [line[0] for line in lines]
    ids = [line[1] for line in lines]
    labels = [line[2] for line in lines]

    return lines, centers, ids, labels


def sort_files(dirpath, metadata_path):
    # read all files in dirpath (including subdirectories) that ends with .nii.gz
    print(f'Reading files from {dirpath}')
    all_files = []
    for root, _, filenames in os.walk(dirpath):
        for filename in tqdm(filenames, desc='Reading files'):
            if filename.endswith('.nii.gz'):
                all_files.append(os.path.join(root, filename))

    # read metadata file
    print(f'Reading metadata from {metadata_path}')
    metadata = {}
    _, centers, ids, _ = get_data(metadata_path)
    for center, id in zip(centers, ids):
        metadata[id] = center

    # create directories for each center
    print(f'Creating directories for {len(set(metadata.values()))} centers')
    for center in set(metadata.values()):
        os.makedirs(os.path.join(dirpath, center), exist_ok=True)

    # sort files by center name
    print(f'Sorting {len(all_files)} files')
    for file in tqdm(all_files, desc='Sorting files'):
        file_id = os.path.basename(file)[:-7]
        center = metadata[file_id]
        # copy file to the corresponding center directory
        shutil.copy(file, os.path.join(dirpath, center, os.path.basename(file)))


if __name__ == '__main__':
    dirpath = '/tanghaomiao/medai/data'
    metadata_path = '/tanghaomiao/medai/label4Tang.csv'
    sort_files(dirpath, metadata_path)
