import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SAMPLE_PATH = '/tanghaomiao/medai/data/train/ZR/201601180059.nii.gz'

img = nib.load(SAMPLE_PATH)
data = img.get_fdata()

slice_idx = 50  # Choose a slice index

slice = data[:, :, slice_idx]

slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice)) * 255

# show non-zero values in data slice
non_zero_values = np.nonzero(slice)
print(non_zero_values)

Image.fromarray(slice).convert('RGB').save('test_pillow.png')

plt.imshow(slice, cmap='gray')
plt.savefig('test_matplotlib.png')
