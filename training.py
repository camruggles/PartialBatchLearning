# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pdb
import time
import sys
import random

from resnet import resnet34, ResNet

t1 = time.time()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# create reproducible results
# seed = 117 #seed
# torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# random.seed(seed)

# Hyper-parameters
torch.backends.cudnn.benchmark = True
num_epochs = 10
learning_rate = 0.1

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self , transformation=transforms.ToTensor()):
        self.cifar10 = datasets.CIFAR10(root='./data',
                                        download=True,
                                        train=True,
                                        transform=transformation)
        
    def __getitem__(self, index):
        data, target = self.cifar10.__getitem__(index)
        
        # Your transformations here (or set it in CIFAR10)
        
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

train_dataset = MyDataset(transform)


test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            download=True,
                                            train=False,
                                            transform=transforms.ToTensor()#transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                                            )

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)


model = resnet34().to(device)
print(len(train_dataset))
model.setFeatureBankSize(len(train_dataset)) #Cameron

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)#, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train_loop():
    for i, (images, labels, idx) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        
        labels = labels.to(device)
        idx = idx.to(device)
        # Forward pass
        outputs = model((images,idx))
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


def test_loop():
    idx = torch.zeros(1)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            idx = idx.to(device)
            outputs = model((images, idx))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        
    model.train()


if True:
    # Train the model
    model.train()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        sys.stderr.write("Epoch %d\n" % (epoch))
        te = time.time()

        train_loop()

        print("Time for epoch {} : {}".format(epoch+1, time.time()-te))
        
        if (epoch+1) % 10 == 0:
            test_loop()
    scheduler.step()

    # torch.save(model.state_dict(), 'resnet_60.ckpt')
    # quit()
else:
    model.load_state_dict(torch.load("resnet.ckpt"))
print(time.time()-t1)

num_epochs = 200

model.setConstruction(True)
# Construct SUFB
model.eval()
with torch.no_grad():
    for i, (images, labels, idx) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        idx = idx.to(device)

        # Forward pass
        outputs = model((images, idx))

model.setConstruction(False)
model.preparePhaseTwo()

t1 = time.time()
# Train the model phase two

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


model.train()
model.setPhaseTwo(True)


total_step = len(train_loader)
for epoch in range(num_epochs):
    sys.stderr.write("Epoch %d\n" % (epoch+1))
    te=time.time()
    
    train_loop()
    print("Time for epoch {} : {} ".format(epoch+1, time.time() - te))

    if (epoch+1) % 2 == 0:
        # Test the model
        model.setPhaseTwo(False)
        test_loop()
        model.setPhaseTwo(True)
    
    scheduler.step()

print(time.time()-t1)
model.deactivatePhaseTwo()


# Test the model
test_loop()

# torch.save(model.state_dict(), 'resnet2p100.ckpt')


# Save the model checkpoint
# torch.save(model.state_dict(), 'resnet.ckpt')
