import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
#import pickle
from PIL import Image

class Enc_Dec(nn.Module):

	def __init__(self, input_chanels, n_featers):
		super().__init__()

		self.conv1_1 = nn.Conv2d(input_chanels, n_featers, kernel_size = 5, stride = 1, padding = 2)
		self.conv1_2 = nn.Conv2d(n_featers, n_featers, kernel_size = 5 , stride = 1 , padding = 2)
		self.conv1_3 = nn.Conv2d(n_featers, n_featers*2, kernel_size = 5 , stride = 1 , padding = 2)
		
		self.pool1 = nn.MaxPool2d(kernel_size = [3, 4], stride = [3, 4], return_indices = True)

		self.conv2_1 = nn.Conv2d(n_featers*2, n_featers*2, kernel_size = 5, stride = 1, padding = 2)
		self.conv2_2 = nn.Conv2d(n_featers*2, n_featers*2, kernel_size = 5 , stride = 1 , padding = 2)
		self.conv2_3 = nn.Conv2d(n_featers*2, n_featers*3, kernel_size = 5 , stride = 1 , padding = 2)

		self.pool2 = nn.MaxPool2d(kernel_size = [3, 4], stride = [3, 4], return_indices = True)

		self.conv3_1 = nn.Conv2d(n_featers*3, n_featers*3, kernel_size = 5, stride = 1, padding = 2)
		self.conv3_2 = nn.Conv2d(n_featers*3, n_featers*3, kernel_size = 5 , stride = 1 , padding = 2)
		self.conv3_3 = nn.Conv2d(n_featers*3, n_featers*4, kernel_size = 5 , stride = 1 , padding = 2)

		self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)

		self.conv4_1 = nn.Conv2d(n_featers*4, n_featers*4, kernel_size = 5, stride = 1, padding = 2)
		self.conv4_2 = nn.Conv2d(n_featers*4, n_featers*5, kernel_size = 5 , stride = 1 , padding = 2)

		self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)

		self.conv5_1 = nn.Conv2d(n_featers*5, n_featers*5, kernel_size = 3, stride = 1, padding = 0)
		self.conv5_2 = nn.Conv2d(n_featers*5, n_featers*5, kernel_size = 3, stride = 1, padding = 0)
		self.conv5_3 = nn.Conv2d(n_featers*5, n_featers*10, kernel_size = 3, stride = 1, padding = 0)



		########################################## UPSAMPLING ###############################################



		self.conv5_3r = nn.ConvTranspose2d(n_featers*10, n_featers*5, kernel_size = 3, stride = 1, padding = 0)
		self.conv5_2r = nn.ConvTranspose2d(n_featers*5, n_featers*5, kernel_size = 3, stride = 1, padding = 0)
		self.conv5_1r = nn.ConvTranspose2d(n_featers*5, n_featers*5, kernel_size = 3, stride = 1, padding = 0)

		self.unpool4 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
		self.last_conv4 = nn.Conv2d(n_featers*5, n_featers*5, kernel_size = 5, stride = 1, padding = 2)

		self.conv4_2r = nn.ConvTranspose2d(n_featers*5, n_featers*4, kernel_size = 5, stride = 1, padding = 2)
		self.conv4_1r = nn.ConvTranspose2d(n_featers*4, n_featers*4, kernel_size = 5, stride = 1, padding = 2)
		
		self.unpool3 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
		self.last_conv3 = nn.Conv2d(n_featers*4, n_featers*4, kernel_size = 5, stride = 1, padding = 2)

		self.conv3_3r = nn.ConvTranspose2d(n_featers*4, n_featers*3, kernel_size = 5, stride = 1, padding = 2)
		self.conv3_2r = nn.Conv2d(n_featers*3, n_featers*3, kernel_size = 5, stride = 1, padding = 2)
		self.conv3_1r = nn.Conv2d(n_featers*3, n_featers*3, kernel_size = 5, stride = 1, padding = 2)

		self.unpool2 = nn.MaxUnpool2d(kernel_size = [3, 4], stride = [3, 4])
		self.last_conv2 = nn.Conv2d(n_featers*3, n_featers*3, kernel_size = 5, stride = 1, padding = 2)

		self.conv2_3r = nn.ConvTranspose2d(n_featers*3, n_featers*2, kernel_size = 5, stride = 1, padding = 2)
		self.conv2_2r = nn.Conv2d(n_featers*2, n_featers*2, kernel_size = 5, stride = 1, padding = 2)
		self.conv2_1r = nn.Conv2d(n_featers*2, n_featers*2, kernel_size = 5, stride = 1, padding = 2)

		self.unpool1 = nn.MaxUnpool2d(kernel_size = [3, 4], stride = [3, 4])
		self.last_conv1 = nn.Conv2d(n_featers*2, n_featers*2, kernel_size = 5, stride = 1, padding = 2)

		self.conv1_3r = nn.ConvTranspose2d(n_featers*2, n_featers, kernel_size = 5, stride = 1, padding = 2)
		self.conv1_2r = nn.Conv2d(n_featers, n_featers, kernel_size = 5, stride = 1, padding = 2)
		self.conv1_1r = nn.Conv2d(n_featers, input_chanels, kernel_size = 5, stride = 1, padding = 2)


	def forward(self, x):

		#global flag

		#plt.imshow(x.cpu().clone().detach().numpy()[0][0])

		#plt.show()
		#self.conv1_1r.weight = self.conv1_1.weight
		#self.pool1.weight = self.unpool1.weight
		self.conv2_1r.weight = self.conv2_1.weight
		#self.pool2.weight = self.unpool2.weight
		self.conv3_1r.weight = self.conv3_1.weight
		#self.pool3.weight = self.unpool3.weight
		self.conv4_1r.weight = self.conv4_1.weight
		self.conv5_1r.weight = self.conv5_1.weight

		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = F.relu(self.conv1_3(x))

		x, ind1 = self.pool1(x)
		
		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = F.relu(self.conv2_3(x))
		
		x, ind2 = self.pool2(x)

		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = F.relu(self.conv3_3(x))

		x, ind3 = self.pool3(x)

		x = F.relu(self.conv4_1(x))
		x = F.relu(self.conv4_2(x))	

		x, ind4 = self.pool4(x)

		x = F.relu(self.conv5_1(x))
		x = F.relu(self.conv5_2(x))
		x = F.relu(self.conv5_3(x))

		################## UPSAMPLING #########################

		x = F.relu(self.conv5_3r(x))
		x = F.relu(self.conv5_2r(x))
		x = F.relu(self.conv5_1r(x))

		x = self.unpool4(x, ind4)
		x = F.relu(self.last_conv4(x))

		x = F.relu(self.conv4_2r(x))
		x = F.relu(self.conv4_1r(x))
		
		x = self.unpool3(x, ind3)
		x = F.relu(self.last_conv3(x))

		x = F.relu(self.conv3_3r(x))
		x = F.relu(self.conv3_2r(x))
		x = F.relu(self.conv3_1r(x))

		x = self.unpool2(x, ind2)
		x = F.relu(self.last_conv2(x))

		x = F.relu(self.conv2_3r(x))
		x = F.relu(self.conv2_2r(x))
		x = F.relu(self.conv2_1r(x))

		x = self.unpool1(x, ind1)
		x = F.relu(self.last_conv1(x))

		x = F.relu(self.conv1_3r(x))
		x = F.relu(self.conv1_2r(x))
		x = F.relu(self.conv1_1r(x))
	
		return x, [ind1, ind2, ind3, ind4]


class Decoder(nn.Module):

	def __init__(self, input_chanels, n_featers):
		super().__init__()

		self.conv5_3r = nn.ConvTranspose2d(n_featers*10, n_featers*5, kernel_size = 3, stride = 1, padding = 0)
		self.conv5_2r = nn.ConvTranspose2d(n_featers*5, n_featers*5, kernel_size = 3, stride = 1, padding = 0)
		self.conv5_1r = nn.ConvTranspose2d(n_featers*5, n_featers*5, kernel_size = 3, stride = 1, padding = 0)

		self.unpool4 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
		self.last_conv4 = nn.Conv2d(n_featers*5, n_featers*5, kernel_size = 5, stride = 1, padding = 2)

		self.conv4_2r = nn.ConvTranspose2d(n_featers*5, n_featers*4, kernel_size = 5, stride = 1, padding = 2)
		self.conv4_1r = nn.ConvTranspose2d(n_featers*4, n_featers*4, kernel_size = 5, stride = 1, padding = 2)
		
		self.unpool3 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
		self.last_conv3 = nn.Conv2d(n_featers*4, n_featers*4, kernel_size = 5, stride = 1, padding = 2)

		self.conv3_3r = nn.ConvTranspose2d(n_featers*4, n_featers*3, kernel_size = 5, stride = 1, padding = 2)
		self.conv3_2r = nn.Conv2d(n_featers*3, n_featers*3, kernel_size = 5, stride = 1, padding = 2)
		self.conv3_1r = nn.Conv2d(n_featers*3, n_featers*3, kernel_size = 5, stride = 1, padding = 2)

		self.unpool2 = nn.MaxUnpool2d(kernel_size = [3, 4], stride = [3, 4])
		self.last_conv2 = nn.Conv2d(n_featers*3, n_featers*3, kernel_size = 5, stride = 1, padding = 2)

		self.conv2_3r = nn.ConvTranspose2d(n_featers*3, n_featers*2, kernel_size = 5, stride = 1, padding = 2)
		self.conv2_2r = nn.Conv2d(n_featers*2, n_featers*2, kernel_size = 5, stride = 1, padding = 2)
		self.conv2_1r = nn.Conv2d(n_featers*2, n_featers*2, kernel_size = 5, stride = 1, padding = 2)

		self.unpool1 = nn.MaxUnpool2d(kernel_size = [3, 4], stride = [3, 4])
		self.last_conv1 = nn.Conv2d(n_featers*2, n_featers*2, kernel_size = 5, stride = 1, padding = 2)

		self.conv1_3r = nn.ConvTranspose2d(n_featers*2, n_featers, kernel_size = 5, stride = 1, padding = 2)
		self.conv1_2r = nn.Conv2d(n_featers, n_featers, kernel_size = 5, stride = 1, padding = 2)
		self.conv1_1r = nn.Conv2d(n_featers, input_chanels, kernel_size = 5, stride = 1, padding = 2)

		self.linear_exp1 = nn.Linear(input_chanels*640*360, input_chanels*640*360)


	def forward(self, x, indices):

		ind1, ind2, ind3, ind4 = indices

		x = F.relu(self.conv5_3r(x))
		x = F.relu(self.conv5_2r(x))
		x = F.relu(self.conv5_1r(x))

		x = self.unpool4(x, ind4)
		x = F.relu(self.last_conv4(x))

		x = F.relu(self.conv4_2r(x))
		x = F.relu(self.conv4_1r(x))
		
		x = self.unpool3(x, ind3)
		x = F.relu(self.last_conv3(x))

		x = F.relu(self.conv3_3r(x))
		x = F.relu(self.conv3_2r(x))
		x = F.relu(self.conv3_1r(x))

		x = self.unpool2(x, ind2)
		x = F.relu(self.last_conv2(x))

		x = F.relu(self.conv2_3r(x))
		x = F.relu(self.conv2_2r(x))
		x = F.relu(self.conv2_1r(x))

		x = self.unpool1(x, ind1)
		x = F.relu(self.last_conv1(x))

		x = F.relu(self.conv1_3r(x))
		x = F.relu(self.conv1_2r(x))
		x = F.relu(self.conv1_1r(x))

		return x









