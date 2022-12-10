from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
from tps_transformation import tps_transform


class CustomDataset(Dataset):
    def __init__(self, sketch_dir, reference_dir, s_transform=None, r_transform=None):
        super(CustomDataset, self).__init__()
        self.sketch_dir = sketch_dir
        self.reference_dir = reference_dir
        self.sketch_filename = [x for x in sorted(os.listdir(self.sketch_dir))]
        self.reference_filename = [x for x in sorted(os.listdir(self.reference_dir))]
        self.s_transform = s_transform
        self.r_transform = r_transform

    def __getitem__(self, index):
        sketch_path = os.path.join(self.sketch_dir, self.sketch_filename[index])
        reference_path = os.path.join(self.reference_dir, self.reference_filename[index])
        sketch = Image.open(sketch_path)
        reference = Image.open(reference_path)
        reference = np.array(reference)
        # noise = np.random.uniform(-1, 1, np.shape(reference))
        # reference = reference + noise
        reference_tps = tps_transform(reference)
        reference = Image.fromarray(reference.astype('uint8'))
        reference_tps = Image.fromarray(reference_tps.astype('uint8'))

        sketch = self.s_transform(sketch)
        reference = self.r_transform(reference)
        reference_tps = self.r_transform(reference_tps)
        return sketch, reference_tps, reference

    def __len__(self):
        return len(self.sketch_filename)


