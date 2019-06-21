import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import pickle
import utils
import os.path
from os import listdir
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
from utils import UnsuperviseDataset, train
from utils import DenseAutoencoder
from utils import ConvAutoencoder
from utils import ConvAutoencoder_dense_out
from utils import ConvAutoencoder_conv1x1
from utils import ConvAutoencoder_conv1x1_layertest
from utils import ConvAutoencoder_deeper1
from helper import imshow, list_plot
from dataset_statistics import dataSetStatistics


data_dir = '/home/jfeinst/Projects/bionoi_autoencoder_modified/test/'
img_list = []
for item in listdir(data_dir):
    if isfile(join(data_dir, item)):
        img_list.append(item)
    elif isdir(join(data_dir, item)):
        update_data_dir = join(data_dir, item)
        for f in listdir(update_data_dir):
            if isfile(join(update_data_dir, f)):
                img_list.append(f)
print(len(img_list))