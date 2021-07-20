import torch
from resnet import resnet18

# print('Hello world')
# print(torch.cuda.is_available())










# Image preprocessing modules
transformation = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self , transformation):
        self.cifar10 = datasets.CIFAR10(root='YOUR_PATH',
                                        download=False,
                                        train=True,
                                        transform=transformation)
        
    def __getitem__(self, index):
        data, target = self.cifar10.__getitem__(index)
        
        # Your transformations here (or set it in CIFAR10)
        
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

dataset = MyDataset(transform)