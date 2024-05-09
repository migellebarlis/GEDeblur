import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from gopro import GoProDataset
from hide import HIDEDataset
from realblurr import RealBlurRDataset
from realblurj import RealBlurJDataset

class DeblurringDataLoader():
    def __init__(self, set_name, image_size=256, batch_size=100, num_workers=4):
        if set_name == 'gopro':
            path = 'G:/GOPRO_Large'
        elif set_name in ['realblurr', 'realblurj']:
            path = 'G:/RealBlur'
        elif set_name == 'hide':
            path = 'G:/HIDE'

        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        valid_transform = transforms.Compose([
            transforms.CenterCrop(image_size),
        ])

        if set_name == 'gopro':
            train_set = GoProDataset(path, train=True, transform=train_transform)
            valid_set = GoProDataset(path, train=False, transform=valid_transform)
        elif set_name == 'realblurr':
            train_set = RealBlurRDataset(path, train=True, transform=train_transform)
            valid_set = RealBlurRDataset(path, train=False, transform=valid_transform)
        elif set_name == 'realblurj':
            train_set = RealBlurJDataset(path, train=True, transform=train_transform)
            valid_set = RealBlurJDataset(path, train=False, transform=valid_transform)
        elif set_name == 'hide':
            train_set = HIDEDataset(path, train=True, transform=train_transform)
            valid_set = HIDEDataset(path, train=False, transform=valid_transform)

        self.train_dl = DataLoader(train_set, batch_size=batch_size, drop_last=True, shuffle=False, pin_memory=True, num_workers=num_workers, persistent_workers=True)
        self.valid_dl = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, persistent_workers=True)