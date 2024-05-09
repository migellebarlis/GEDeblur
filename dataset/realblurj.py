import os

import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

class RealBlurJDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        # List all the active directories
        self.image_pairs = []
        self.path_pairs = []
        
        # Take files from list
        with open('%s/%s'%(path,'RealBlur_J_'+('train_list.txt' if train == True else 'test_list.txt')),'rb') as f:
            dirs = [line.strip().decode() for line in f]
            dirs = [d.split(' ') for d in dirs]

        for dir in dirs:
            input = os.path.join(path, dir[1])
            target = os.path.join(path, dir[0])

            self.image_pairs.append([input, target])
            self.path_pairs.append([dir[1], dir[0]])

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