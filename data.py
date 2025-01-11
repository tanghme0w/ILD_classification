# dataloader for CT classification
import os
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPImageProcessor
import numpy as np
from PIL import Image

def get_data_dict(filepath):
    """ read csv file and return a dictionary with the number of lines, words, and characters

    Args:
        filepath (str): the path to the csv file

    Returns:
        a dictionary that maps each id to its corresponding center and label
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

    return dict(zip(ids, zip(centers, labels)))

class CTDataset(Dataset):
    def __init__(self, data_dir, metadata_file, processor, num_frames=10):
        self.data_dir = data_dir
        self.files = []
        self.metadata = get_data_dict(metadata_file)
        self.processor = processor
        self.num_frames = num_frames
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.nii.gz'):
                    self.files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # open the file and return the data
        center, label = self.metadata[os.path.basename(self.files[idx]).replace('.nii.gz', '')]
        data = nib.load(self.files[idx]).get_fdata()
        processed_data = []
        sampled_frames = np.random.choice(data.shape[-1], size=min(self.num_frames, data.shape[-1]), replace=False)
        for i in sampled_frames:
            data_slice = data[:, :, i]
            if np.max(data_slice) - np.min(data_slice) == 0:
                continue # skip empty slices
            data_slice = (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice)) * 255
            image = Image.fromarray(data_slice).convert('RGB')
            processed_data.append(self.processor.preprocess(image, return_tensors='pt')['pixel_values'])
        processed_data = torch.cat(processed_data, dim=0) # (N, 3, 224, 224)
        broadcasted_label = torch.tensor(int(label)).expand(processed_data.shape[0]) # (N, )
        return processed_data, broadcasted_label


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    return images, labels


if __name__ == '__main__':
    data_dir = '/tanghaomiao/medai/data'
    metadata_file = '/tanghaomiao/medai/label4Tang.csv'
    processor = CLIPImageProcessor.from_pretrained('/tanghaomiao/medai/clip-vit-large-patch14', local_files_only=True)
    dataset = CTDataset(data_dir, metadata_file, processor)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for data in dataloader:
        print(data.shape)
        break
