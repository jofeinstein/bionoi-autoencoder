"""
Given the index of an image, take this image and feed it to a trained autoencoder,
then plot the original image and reconstructed image.
This code is used to visually verify the correctness of the autoencoder
"""
import torch
import argparse
import pickle
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
from os import listdir
from os.path import isfile, join
from utils import UnsuperviseDataset, ConvAutoencoder_conv1x1
from helper import imshow
from dataset_statistics import dataSetStatistics

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-index',
                        default=0,
                        type=int,
                        required=False,
                        help='index of image')
    parser.add_argument('-data_dir',
                        default='../bindingsite-img/',
                        required=False,
                        help='directory of training images')
    parser.add_argument('-feature_dir',
                        default='../bindingsite-feature/',
                        required=False,
                        help='directory of generated features')                      
    parser.add_argument('-normalize',
                        default=True,
                        required=False,
                        help='whether to normalize dataset')
    parser.add_argument('-num_data',
                        type=int,
                        default=50000,
                        required=False,
                        help='the batch size, normally 2^n.')
    return parser.parse_args()

def feature_vec_gen(device, model, dataset, feature_dir):
    """
    Generate feature vectors for a single image
    """
    m = len(dataset)
    for i in range(m):
        image, file_name = dataset[i]
        file_name = file_name.split('.')
        file_name = file_name[0]
        image = image.to(device) # send to gpu
        feature_vec = model.encode_vec(image.unsqueeze(0)) # add one dimension
        #print('shape of feature vector:', feature_vec.shape)
        pickle_file = open(feature_dir + file_name + '.pickle','wb')
        pickle.dump(feature_vec, pickle_file)

if __name__ == "__main__":
    args = getArgs()
    index = args.index
    data_dir = args.data_dir
    feature_dir = args.feature_dir
    normalize = args.normalize
    num_data = args.num_data

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))   

    # model configuration
    model = ConvAutoencoder_conv1x1()
    model_file = './log/conv1x1norm.pt'
    # if there are multiple GPUs, split a batch to different GPUs
    if torch.cuda.device_count() > 1:
        print("Using "+str(torch.cuda.device_count())+" GPUs...")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))
    print(model)
    #--------------------model configuration ends--------------------------

    # data configuration
    data_mean = dataSetStatistics(data_dir, 128, num_data)[0].tolist()
    data_std = dataSetStatistics(data_dir, 128, num_data)[1].tolist()
    if normalize == True:
        print('normalizing data:')
        print('mean:', data_mean)
        print('std:', data_std)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((data_mean[0], data_mean[1], data_mean[2]),
                                                             (data_std[0], data_std[1], data_std[2]))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    img_list = [f for f in listdir(data_dir)] # put images into dataset
    dataset = UnsuperviseDataset(data_dir, img_list, transform=transform)  
    image, file_name = dataset[index]
    # calulate the input size (flattened)
    print('name of input:', file_name)
    image_shape = image.shape
    print('shape of input:', image_shape)

    # generate features for images in data_dir
    model = model.to(device)
    model.eval() # don't cache the intermediate values
    print('generating feature vectors...')
    feature_vec_gen(device, model, dataset, feature_dir)