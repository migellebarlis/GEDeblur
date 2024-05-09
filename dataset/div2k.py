import os

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

class DIV2KDataset(Dataset):
    def __init__(self, path):
        # List all the active directories
        self.image_pairs = []
        self.path_pairs = []
        
        for dir in os.scandir(os.path.join(path, 'blur')):
            for file in os.scandir(dir):
                target = os.path.join(path, 'sharp', file.name)

                if os.path.exists(target):
                    self.image_pairs.append([file.path, target])
                    self.path_pairs.append(['blur/'+dir.name+'/'+file.name, 'sharp/'+file.name])

    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tensor = transforms.ToTensor()
        
        input = tensor(Image.open(self.image_pairs[idx][0]))
        target = tensor(Image.open(self.image_pairs[idx][1]))

        return input, target, self.path_pairs[idx][0], self.path_pairs[idx][1]