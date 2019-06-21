import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data
import copy
import argparse
from os import listdir
from os.path import isfile, join, isdir
from utils import UnsuperviseDataset


data_dir = '/home/jfeinst/Projects/bionoi_autoencoder_modified/test/'
img_list = []
for item in listdir(data_dir):
    if isfile(join(data_dir, item)):
        img_list.append(item)
    elif isdir(join(data_dir, item)):
        update_data_dir = join(data_dir, item)
        for f in listdir(update_data_dir):
            if isfile(join(update_data_dir, f)):
                img_list.append(item+'/'+f)


def dataSetStatistics(data_dir, batch_size, num_data):
    # Detect if we have a GPU available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('Current device: '+str(device))

    transform = transforms.Compose([transforms.ToTensor()])
    #img_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

    img_list = []
    for item in listdir(data_dir):
        if isfile(join(data_dir, item)):
            img_list.append(item)
        elif isdir(join(data_dir, item)):
            update_data_dir = join(data_dir, item)
            for f in listdir(update_data_dir):
                if isfile(join(update_data_dir, f)):
                    img_list.append(item + '/' + f)

    dataset = UnsuperviseDataset(data_dir, img_list, transform=transform)
    total = dataset.__len__()
    print('length of entire dataset:', total)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    # calculate mean and std for training data
    mean = 0.
    std = 0.
    m = 0
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1) # reshape
        mean = mean + data.mean(2).sum(0)
        std = std + data.std(2).sum(0)
        m = m + batch_samples
        if m > num_data:
            break
    mean = mean / m
    std = std / m
    #print('mean:',mean)
    #print('std:',std)
    return mean, std

print(dataSetStatistics(data_dir, 128, 50000))