"""
Train an autoencoder for bionoi datasets.
User can specify the types (dense-style or cnn-style) and options (denoising, sparse, contractive).
"""
import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import pickle
import utils
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt 
from utils import UnsuperviseDataset, DenseAutoencoder, train
from helper import imshow, list_plot

def getArgs():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-seed',
                        default=123,
						type=int,
                        required=False,
                        help='seed for random number generation')

    parser.add_argument('-epoch',
                        default=8,
						type=int,
                        required=False,
                        help='number of epochs to train')

    parser.add_argument('-feature_size',
                        default=256,
						type=int,
                        required=False,
                        help='size of output feature of the autoencoder')

    parser.add_argument('-data_dir',
                        default='../bae-data-images/',
                        required=False,
                        help='directory of training images')

    parser.add_argument('-feature_dir',
                        default='../bae-data-features/dense/',
                        required=False,
                        help='directory of output features stored in pickle file')

    parser.add_argument('-model_file',
                        default='../trained_model/bionoi_autoencoder_dense.pt',
                        required=False,
                        help='file to save the model')

    parser.add_argument('-log_dir',
                        default='./log/',
                        required=False,
                        help='directory of training images')						

    parser.add_argument('-batch_size',
                        default=512,
						type=int,
                        required=False,
                        help='the batch size, normally 2^n.')

    parser.add_argument('-normalize',
                        default=True,
                        required=False,
                        help='whether to normalize dataset')

    return parser.parse_args()

if __name__ == "__main__":
	args = getArgs()
	seed = args.seed
	num_epochs = args.epoch
	data_dir = args.data_dir
	feature_dir = args.feature_dir
	model_file = args.model_file
	log_dir = args.log_dir
	batch_size = args.batch_size
	normalize = args.normalize
	feature_size = args.feature_size
	
	# Detect if we have a GPU available
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Current device: '+str(device))

	 # define a transform with mean and std, another transform without them.
	data_mean = [0.6150, 0.4381, 0.6450]
	data_std = [0.6150, 0.4381, 0.6450]   
    if normalize == True:
        transformWithNorm = transforms.Compose([
	                        transforms.ToTensor(),
	                        transforms.Normalize((data_mean[0], data_mean[1], data_mean[2]),
	                                             (data_std[0], data_std[1], data_std[2]))
	                                           ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
		
	# put images into dataset
	img_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
	dataset = UnsuperviseDataset(data_dir, img_list, transform=transform)  
	
	# create dataloader
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
	
	# get some random training images to show
	dataiter = iter(dataloader)
	images, filename = dataiter.next()

	#print(images.shape)
	#imshow(torchvision.utils.make_grid(images))
	#images = images.view(batch_size,-1)
	#images = torch.reshape(images,(images.size(0),3,256,256))
	#imshow(torchvision.utils.make_grid(images))
	
	# calulate the input size (flattened)
	image_shape = images.shape
	print('shape of input:', image_shape)
	input_size = image_shape[1]*image_shape[2]*image_shape[3]
	print('flattened input size:',input_size) 

	# instantiate model
	model = DenseAutoencoder(input_size, feature_size)
	# if there are multiple GPUs, split the batch to different GPUs
	if torch.cuda.device_count() > 1:
		print("Using "+str(torch.cuda.device_count())+" GPUs...")
		model = nn.DataParallel(model)

	# print the paramters to train
	print('paramters to train:')
	for name,param in model.named_parameters():
		if param.requires_grad == True:
			print(str(name))

	# loss function  
	criterion = nn.MSELoss()

	# optimizer  
	optimizer = optim.Adam(model.parameters(), 
						   lr=0.001, 
						   betas=(0.9, 0.999), 
						   eps=1e-08, 
						   weight_decay=0.00001, 
						   amsgrad=False)

	learningRateScheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
														   milestones=[2,5,8], 
														   gamma=0.2)		

	# begin training 
	trained_model, train_loss_history = train(device=device, 
											  num_epochs=num_epochs, 
											  dataloader=dataloader, 
											  model=model, 
											  criterion=criterion, 
											  optimizer=optimizer, 
											  learningRateScheduler=learningRateScheduler)
	
	
	# save the model
	torch.save(trained_model.state_dict(), model_file)
	
	# plot the loss function vs epoch
	list_plot(train_loss_history)
	plt.show()