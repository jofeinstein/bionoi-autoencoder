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
import os
from os import listdir
from os.path import isfile, join, isdir
from utils import UnsuperviseDataset, ConvAutoencoder_conv1x1
from helper import imshow
from dataset_statistics import dataSetStatistics
import tarfile
import shutil

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-index',
                        default=0,
                        type=int,
                        required=False,
                        help='index of image')
    parser.add_argument('-data_dir',
                        default='/work/jfeins1/resnet-binary-stripped',
                        required=False,
                        help='directory of training images')
    parser.add_argument('-feature_dir',
                        default='/work/jfeins1/features-binary/',
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
    parser.add_argument('-model',
                        default=ConvAutoencoder_conv1x1(),
                        required=False,
                        help='the model of autoencoder to generate feature vectors with')
    parser.add_argument('-model_file',
                        default='./conv1x1-4M-batch512.pt',
                        required=False,
                        help='directory of trained model')
    parser.add_argument('-gpu_to_cpu',
                        default=False,
                        required=False,
                        help='whether to reconstruct image using model made with gpu on a cpu')
    parser.add_argument('-tar_dir',
                        default='/work/jfeins1/resnet-binary-stripped.tar.gz',
                        required=False,
                        help='directory of tarfile')

    parser.add_argument('-tar_extract_path',
                        default='/var/scratch/jfeins1/',
                        required=False,
                        help='directory to extract tarfile to')

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
        directory_name = file_name.split('/')[0] + '/' + file_name.split('/')[1]
        image = image.to(device) # send to gpu
        feature_vec = model.module.encode_vec(image.unsqueeze(0)) # add one dimension
        # feature_vec = model.encode_vec(image.unsqueeze(0))  # choose this line if running on cpu only machine
        # print('shape of feature vector:', feature_vec.shape)
        if not os.path.exists(feature_dir + '/' + directory_name):
            os.makedirs(feature_dir + '/' + directory_name)
        pickle_file = open(feature_dir + '/' + file_name + '.pickle', 'wb')
        pickle.dump(feature_vec, pickle_file)

if __name__ == "__main__":
    args = getArgs()
    index = args.index
    # data_dir = args.data_dir
    feature_dir = args.feature_dir
    normalize = args.normalize
    num_data = args.num_data
    model = args.model
    model_file = args.model_file
    gpu_to_cpu = args.gpu_to_cpu
    tar_dir = args.tar_dir
    tar_extract_path = args.tar_extract_path

    fold_lst = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']

    fold = 5
    '''model_file = '/home/jfeinst/Projects/bionoi_autoencoder_modified/conv1x1-4M-batch512.pt'
    #gpu_to_cpu = True
    tar_dir = '/home/jfeinst/Projects/resnet-files/resnet-binary-stripped.tar.gz'
    tar_extract_path = '/home/jfeinst/Desktop/'
    tar_name = tar_dir.split('/')[-1].split('.')[0]
    feature_dir = '/home/jfeinst/Desktop/features-binary/'''

    '''model_file = '/home/jfeinst/Projects/bionoi_autoencoder_modified/conv1x1-4M-batch512.pt'
    gpu_to_cpu = True
    tar_dir = '/home/jfeinst/Projects/resnet-files/resnet-binary-stripped.tar.gz'
    tar_extract_path = '/home/jfeinst/Desktop/'
    tar_name = tar_dir.split('/')[-1].split('.')[0]
    feature_dir = '/home/jfeinst/Desktop/features-test/'''

    tar_name = tar_dir.split('/')[-1].split('.')[0]
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Current device: '+str(device))

    if gpu_to_cpu == True:
        # original saved file with DataParallel
        print('GPU to CPU true...')
        state_dict = torch.load(model_file, map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    else:
        # if there are multiple GPUs, split the batch to different GPUs
        if torch.cuda.device_count() > 1:
            print("Using " + str(torch.cuda.device_count()) + " GPUs...")
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_file))


        #--------------------model configuration ends--------------------------

    for i in range(fold):
        print('\n', '--Fold {}--'.format(i + 1))

        print('Extracting tarball...')

        # Untar tarball containing data
        with tarfile.open(tar_dir) as tar:
            subdir_and_files = [tarinfo for tarinfo in tar.getmembers() if
                                tarinfo.name.startswith(tar_name + '/' + fold_lst[i])]
            tar.extractall(members=subdir_and_files, path=tar_extract_path)

        for phase in ['train', 'val']:

            print(phase, '\n')

            data_dir = tar_extract_path + tar_name + '/' + fold_lst[i] + '/' + phase + '/'

            # data configuration
            statistics = dataSetStatistics(data_dir, 128, num_data)
            data_mean = statistics[0].tolist()
            data_std = statistics[1].tolist()

            # data_mean = [0.5834683179855347, 0.6131182312965393, 0.543856143951416]
            # data_std = [0.08377586305141449, 0.09421063959598541, 0.09517180174589157]

            if normalize == True:
                print('normalizing data:')
                print('mean:', data_mean)
                print('std:', data_std)
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((data_mean[0], data_mean[1], data_mean[2]),
                                                                     (data_std[0], data_std[1], data_std[2]))])
            else:
                transform = transforms.Compose([transforms.ToTensor()])

            img_list = []
            for item in listdir(data_dir):  # /var/scratch/jfeins1/resnet-binary/fold0/train/    item= 1 or 3
                if isfile(join(data_dir, item)):  # /var/scratch/jfeins1/resnet-binary/fold0/train/1/ FALSE
                    img_list.append(item)
                elif isdir(join(data_dir, item)):  # /var/scratch/jfeins1/resnet-binary/fold0/train/1/ TRUE
                    update_data_dir = join(data_dir, item)
                    for f in listdir( update_data_dir):  # /var/scratch/jfeins1/resnet-binary/fold0/train/1/    f= 5iune00 or 3ir5a00
                        if isfile(join(update_data_dir, f)):  # /var/scratch/jfeins1/resnet-binary/fold0/train/1/5iune00 FALSE
                            img_list.append(item + '/' + f)
                        elif isdir(join(update_data_dir, f)):  # /var/scratch/jfeins1/resnet-binary/fold0/train/1/5iune00 TRUE
                            deeper_data_dir = join(update_data_dir, f)  # deeper = /var/scratch/jfeins1/resnet-binary/fold0/train/1/5iune00
                            for y in listdir(deeper_data_dir):
                                if isfile(join(deeper_data_dir, y)):
                                    img_list.append(item + '/' + f + '/' + y)

            if not os.path.exists(feature_dir + fold_lst[i]):
                os.makedirs(feature_dir + fold_lst[i])

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
            feature_vec_gen(device, model, dataset, feature_dir + fold_lst[i] + '/' + phase)

        # Delete directory to make room for next fold
        shutil.rmtree(tar_extract_path + tar_name)
