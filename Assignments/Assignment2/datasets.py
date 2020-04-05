from __future__ import print_function
from PIL import Image
import glob
from tqdm.notebook import tqdm
import torch.utils.data as data

class ImageNetLimited(data.Dataset):
    """ImageNet Limited dataset."""
    
    def __init__(self, root_dir, transform=None):
        data = []
        for sample in tqdm(root_dir):
            data.append(sample)
        self.n = len(root_dir)
        self.data = data

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx]
        