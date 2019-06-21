"""
Given the index of an image, take this image and feed it to a trained autoencoder,
then plot the original image and reconstructed image.
This code is used to visually verify the correctness of the autoencoder
"""
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
from os import listdir
import random
import os.path
from os.path import isfile, join
from utils import UnsuperviseDataset, inference
from helper import imshow
from utils import DenseAutoencoder 
from utils import ConvAutoencoder
from utils import ConvAutoencoder_dense_out
from utils import ConvAutoencoder_conv1x1
from utils import ConvAutoencoder_conv1x1_layertest
from utils import ConvAutoencoder_deeper1
from dataset_statistics import dataSetStatistics

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-index',
                        default=0,
                        type=int,
                        required=False,
                        help='index of image')
    parser.add_argument('-data_dir',
                        default='../bae-data-images/',
                        required=False,
                        help='directory of training images')
    parser.add_argument('-style',
                        default='conv_1x1',
                        required=False,
                        choices=['conv', 'dense', 'conv_dense_out', 'conv_1x1', 'conv_deeper', 'conv_1x1_test'],
                        help='style of autoencoder')
    parser.add_argument('-feature_size',
                        default=1024,
						type=int,
                        required=False,
                        help='size of output feature of the autoencoder')                                                
    parser.add_argument('-normalize',
                        default=True,
                        required=False,
                        help='whether to normalize dataset')
    parser.add_argument('-model',
                        default='./log/conv_1x1.pt',
                        required=False,
                        help='trained model location')
    parser.add_argument('-num_data',
                        type=int,
                        default=50000,
                        required=False,
                        help='the batch size, normally 2^n.')
    parser.add_argument('-gpu_to_cpu',
                        default=False,
                        required=False,
                        help='whether to reconstruct image using model made with gpu on a cpu')
    parser.add_argument('-rmse',
                        default=True,
                        required=False,
                        help='whether to calculate root mean squared error between reconstructed images and '
                             'original images')
    parser.add_argument('-img_count',
                        type=int,
                        default=10,
                        required=False,
                        help='how many reconstructed images to save and show')
    return parser.parse_args()


if __name__ == "__main__":
    args = getArgs()
    index = args.index
    data_dir = args.data_dir
    style = args.style
    feature_size = args.feature_size
    normalize = args.normalize
    model_file = args.model
    num_data = args.num_data
    gpu_to_cpu = args.gpu_to_cpu
    rmse_bool = args.rmse
    img_count = args.img_count


    # Detect if we have a GPU available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print('Current device: '+str(device))

    # Normalizing data and transforming images to tensors
    statistics = dataSetStatistics(data_dir, 128, num_data)
    data_mean = statistics[0].tolist()
    data_std = statistics[1].tolist()

    if normalize == True:
        print('normalizing data:')
        print('mean:', data_mean)
        print('std:', data_std)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((data_mean[0], data_mean[1], data_mean[2]),
                                                             (data_std[0], data_std[1], data_std[2]))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    # Put images into dataset
    img_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    dataset = UnsuperviseDataset(data_dir, img_list, transform=transform)


    # Instantiate and load model
    if style == 'conv':
        model = ConvAutoencoder()
    elif style == 'dense':
        model = DenseAutoencoder(input_size, feature_size)  
    elif style == 'conv_dense_out':
        model = ConvAutoencoder_dense_out(feature_size)
    elif style == 'conv_1x1':
        model = ConvAutoencoder_conv1x1()
    elif style == 'conv_1x1_test':
        model = ConvAutoencoder_conv1x1_layertest()
    elif style == 'conv_deeper':
        model = ConvAutoencoder_deeper1()


    # Loading the trained model
    # Converting the trained model if trained to be usable on the cpu
    # if gpu is unavailable and the model was trained using gpus
    if gpu_to_cpu == True:
        # original saved file with DataParallel
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
            print("Using "+str(torch.cuda.device_count())+" GPUs...")
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_file))


    # Reconstructing images and plotting the root mean square error
    if rmse_bool == True:
        reconstruction_lst = []
        print('Forming reconstruction images...')
        for i in range(dataset.__len__()):
            image, name = dataset[i]
            reconstruct_image = inference(device, image.unsqueeze(0), model)    # create reconstructed image using model
            recon_detach = reconstruct_image.detach()
            recon_cpu = recon_detach.cpu()
            recon_numpy = recon_cpu.numpy()     # convert image to numpy array for easier calculations
            recon_numpy = np.squeeze(recon_numpy, axis=0)   # remove first dimension of array i.e. (1,3,256,256) -> (3,256,256)
            reconstruction_lst.append(recon_numpy)

        original_lst = []
        file_name_lst = []
        print('Extracting original images...')
        for tensor_name_tuple in dataset:
            og_img = tensor_name_tuple[0].numpy()   # create list of original images as numpy arrays
            original_lst.append(og_img)
            file_name = tensor_name_tuple[1]    # create list of file names
            file_name_lst.append(file_name)

        N = 1
        for dim in original_lst[0].shape:   # determine N for root mean square error
            N *= dim

        print('Calculating root mean squared error...')
        rmse_lst = []
        for i in range(len(original_lst)):
            RMSE = ((np.sum((original_lst[i] - reconstruction_lst[i]) ** 2) / N) ** .5)     # calculate rmse
            rmse_lst.append(RMSE)

        # Plot histogram of root mean squared errors between reconstructed images and original images
        print('Plotting root mean squared error...')
        plt.hist(np.asarray(rmse_lst), bins=30)
        plt.ylabel('Number of Image Pairs')
        plt.xlabel('Root Mean Squared')
        plt.title('RMSE  —  ' + model_file)
        plt.savefig('./images/' + 'rmse.png')
        # plt.show()

        # Plot images before and after reconstruction
        print('Constructing figures before and after reconstruction...')

        # create a random list of indices to select images from dataset
        # to compare to their respective reconstructed images
        random_index_lst = []
        for i in range(img_count):
            random_index_lst.append(random.randint(0, dataset.__len__()-1))


        for index in random_index_lst:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
            ax1.imshow(np.transpose(original_lst[index], (1, 2, 0)))
            ax1.set_title('Original Normalized Image  —  ' + file_name_lst[index])
            ax2.imshow(np.transpose(reconstruction_lst[index], (1, 2, 0)))
            ax2.set_title('Reconstructed Image  —  ' + file_name_lst[index])
            # show both figures
            plt.savefig('./images/' + 'reconstructed_' + file_name_lst[index].split('.')[0] + '.png')
            plt.imshow(np.transpose(original_lst[index], (1, 2, 0)))
            plt.imshow(np.transpose(reconstruction_lst[index], (1, 2, 0)))
            # plt.show()


    else:
        random_index_lst = []
        print('Constructing figures before and after reconstruction...')
        for i in range(img_count):
            random_index_lst.append(random.randint(0, dataset.__len__() - 1))   # create random list of indices

        for index in random_index_lst:
            image, file_name = dataset[index]
            # calulate the input size (flattened)
            # print('name of input:', file_name)
            # image_shape = image.shape
            # print('shape of input:', image_shape)
            # input_size = image_shape[0] * image_shape[1] * image_shape[2]
            # print('flattened input size:', input_size)

            # Get the reconstructed image
            reconstruct_image = inference(device, image.unsqueeze(0), model)
            # print('shape of reconstructed image:', reconstruct_image.shape)
            # print(reconstruct_image)

            # Measure the loss between the 2 images
            criterion = nn.MSELoss()
            loss = criterion(image.unsqueeze(0).cpu(), reconstruct_image.cpu())
            print('loss between before and after:', loss)

            # plot images before and after reconstruction
            fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14, 7))
            ax1.imshow(np.transpose(image.numpy(), (1,2,0)))
            ax1.set_title('Original Image  —  ' + file_name)
            ax2.imshow(np.transpose(reconstruct_image.squeeze().detach().cpu().numpy(),(1,2,0)))
            ax2.set_title('Reconstructed Image  —  ' + file_name)
            # show both figures
            # plt.savefig('./images/' + 'reconstruct_' + filename.split('.')[0] + '.png')
            plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
            plt.imshow(np.transpose(reconstruct_image.squeeze().detach().cpu().numpy(), (1, 2, 0)))
            #plt.show()