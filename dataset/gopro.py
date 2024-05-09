import os

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

class GoProDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        # List all the active directories
        self.image_pairs = []
        self.path_pairs = []
        path = os.path.join(path, 'train' if train else 'test')
        for dir in os.scandir(path):
            # Process each subset of the database
            for file in os.scandir(os.path.join(dir, 'blur')):
                # Check if a training pair exists
                target = os.path.join(dir, 'sharp/' + file.name)

                if os.path.exists(target):
                    self.image_pairs.append([file.path, target])
                    self.path_pairs.append([('train/' if train else 'test/')+dir.name+'/blur/'+file.name, ('train/' if train else 'test/')+dir.name+'/sharp/'+file.name])

        self.is_training = train
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tensor = transforms.ToTensor()
        input = tensor(Image.open(self.image_pairs[idx][0]))
        target = tensor(Image.open(self.image_pairs[idx][1]))

        # Perform joint transforms
        image = torch.concat((input, target), dim=0)
        if self.transform:
            image = self.transform(image)

        input, target = torch.tensor_split(image, 2, dim=0)

        return input, target, self.path_pairs[idx][0], self.path_pairs[idx][1]