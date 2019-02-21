import torch
import torchvision
import torch.nn as nn

import numpy as np

# Load the COCO images
valDir = '../dataset/validation'
valset = torchvision.datasets.ImageFolder(valDir,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(512),
                torchvision.transforms.CenterCrop(512),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((125, 125, 125), (125, 125, 125))
            ]))

# Load the labels
labels = torch.load('validation.pt')
valset_ = [None for _ in range(1000)] # take only 1000 for testing purposes
for i, data in enumerate(valset):
    if i >= 1000: break # break after 1000 for testing purposes
    valset_[i] = (data[0], labels[i, :].clone())
    if i % 200 == 0: print(i/len(labels))

# Create and save DataLoader
valloader = torch.utils.data.DataLoader(valset_, batch_size=32, shuffle=True)
torch.save(valloader, 'valloader.pt')
