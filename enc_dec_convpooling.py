import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
#import pickle
from PIL import Image

lr = 0.001
epochs = 0
flag = False
load_m = True
channels_noise = 1800
batch_size = 1
n_featers = 15
input_chanels = 3
dataset_size = 200
mod_save_path = 'models\\model_alpha_convpooling_overfit_test.pth.tar'

resolutions = {'hd': [720, 1280],
			  'full-hd': [1080, 1920]}
resolution = 'hd'

if resolution == 'hd':
	kernel_sizes = [5, 4, 4, [3, 4], [3, 4]]
elif resolution == 'full-hd':
	kernel_sizes = [[9, 16], 8, 5, 2]
	#kernel_sizes = [5, 4, [3, 4], [3, 4], 3, 2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fixed_noise = torch.randn(1, channels_noise, 1, 1).to(device)

def im_convert(tensor):
	image = tensor.cpu().clone().detach().numpy()
	#clone tensor --> detach it from computations --> transform to numpy
	image = image.squeeze()
	image = image.transpose(1, 2, 0)
	# swap axis from(1,28,28) --> (28,28,1)
	image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
	#denormalize image
	image = image.clip(0, 1)
	#sets image range from 0 to 1
	return image


def save_mod(state, filename = mod_save_path):
	torch.save(state, filename)

def load_mod(checkpoint):
	model.load_state_dict(checkpoint['state_dictionary'])
	#optimizer.load_state_dict(['state_dictionary'])

class Enc_Dec(nn.Module):
	def __init__(self, input_chanels, n_featers, resolution, kernel_sizes):
		super().__init__()
		self.resolution = resolution
		if resolution == 'hd':
			ff = 2
		else:
			ff = 1
		self.pad1 = nn.ZeroPad2d((8, 7, 4, 4))
		self.conv1 = nn.Conv2d(input_chanels, n_featers*3*input_chanels, kernel_size = [9, 16], stride = 1, padding = 0)
		self.pool1 = nn.Conv2d(n_featers*3*input_chanels, n_featers*3*input_chanels, kernel_size = [9, 16], stride = [9, 16], padding = 0)

		self.pad2 = nn.ZeroPad2d((3, 4, 3, 4))
		self.conv2 = nn.Conv2d(n_featers*3*input_chanels, n_featers*6*input_chanels, kernel_size = 8, stride = 1, padding = 0)
		self.pool2 = nn.Conv2d(n_featers*6*input_chanels, n_featers*6*input_chanels, kernel_size = 8, stride = 8, padding = 0)
		
		self.conv3 = nn.Conv2d(n_featers*6*input_chanels, n_featers*12*input_chanels, kernel_size = 5, stride = 1, padding = 2)
		self.pool3 = nn.Conv2d(n_featers*12*input_chanels, n_featers*12*input_chanels, kernel_size = 5, stride = 5, padding = 0)
		
		self.conv4 = nn.Conv2d(n_featers*12*input_chanels, n_featers*24*input_chanels, kernel_size = 2, stride = 2, padding = 0)

		self.conv4_r = nn.ConvTranspose2d(n_featers*24*input_chanels, n_featers*12*input_chanels, kernel_size = 2, stride = 2, padding = 0)

		self.unpool3 = nn.ConvTranspose2d(n_featers*12*input_chanels, n_featers*12*input_chanels, kernel_size = 5, stride = 5, padding = 0)
		self.conv3_r = nn.ConvTranspose2d(n_featers*12*input_chanels, n_featers*6*input_chanels, kernel_size = 5, stride = 1, padding = 2)

		self.unpool2 = nn.ConvTranspose2d(n_featers*6*input_chanels, n_featers*6*input_chanels, kernel_size = 8, stride = 8, padding = 0)
		self.conv2_r = nn.ConvTranspose2d(n_featers*6*input_chanels, n_featers*3*input_chanels, kernel_size = 8, stride = 1, output_padding = 0)
		
		self.unpool1 = nn.ConvTranspose2d(n_featers*3*input_chanels, n_featers*3*input_chanels, kernel_size = [9, 16], stride = [9, 16], padding = 0)
		self.conv1_r = nn.ConvTranspose2d(n_featers*3*input_chanels, input_chanels, kernel_size = [9, 16], stride = 1, padding = 0)



	def forward(self, x):
		#global flag
		#plt.imshow(x.cpu().clone().detach().numpy()[0][0])
		#plt.show()
		x = self.pad1(x)
		x = F.relu(self.conv1(x))
		x = F.relu(self.pool1(x))
		x = self.pad2(x)

		x = F.relu(self.conv2(x))
		x = F.relu(self.pool2(x))

		x = F.relu(self.conv3(x))
		x = F.relu(self.pool3(x))

		x = F.relu(self.conv4(x))
		
		x = F.relu(self.conv4_r(x))

		x = F.relu(self.unpool3(x))
		x = F.relu(self.conv3_r(x))

		x = F.relu(self.unpool2(x))
		x = F.relu(self.conv2_r(x))
		x = x[:, :, :80, :80]	
		x = F.relu(self.unpool1(x))
		x = F.relu(self.conv1_r(x))
		x = x[:, :, :720, :1280]
		#print('Pooling... done', end = '\r')
		'''
		if flag:

			indices = {'ind1': ind1, 'ind2': ind2, 'ind3': ind3, 'ind4': ind4, 'ind5': ind5, 'ind6': ind6, 'ind7': ind7}
			torch.save(indices, 'models\\indices')
			#file = open('models\\indices.txt', 'wb')
			#pickle.dump(indices, file)
			#file.close()
			flag = False
		'''
		return x



training_path = 'D:\\Datasets\\Mountain\\mountain\\'
training_dataset_raw = [Image.open(training_path + 'train (' + str(random.randint(1, 11000)) + ').jpg') for i in range(1, dataset_size)]

transform_train = transforms.Compose([transforms.ToTensor(),
									  transforms.Normalize((0.5, 0.5, 0.5),
														   (0.5, 0.5, 0.5)),
									  ])

for idx, img in enumerate(training_dataset_raw):
	training_dataset_raw[idx] = transform_train(img)
	if idx%100 == 0:
		print(idx, ' images have been converted', end = '\r')

training_dataset = torch.stack(training_dataset_raw)  
print(type(training_dataset))
training_dataset = torch.utils.data.TensorDataset(training_dataset, training_dataset)
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
#torch.reshape(training_dataset, (-1, batch_size))
print(training_dataset)
torch.cuda.empty_cache()

#sys.exit()
criterion = nn.MSELoss(reduction='sum')
model = Enc_Dec(input_chanels, n_featers, 'hd', kernel_sizes).to(device)
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr = lr)

if load_m:
	load_mod(torch.load('models\\model_alpha_convpooling_overfit_test.pth.tar'))



for epoch in range(epochs):
	torch.cuda.empty_cache()
	checkpoint = {'state_dictionary' : model.state_dict(), 'optimizer': optimizer.state_dict()}
	running_loss = 0.0
	if epoch % 1 == 0:
		save_mod(checkpoint)
	for idx, [data, label] in enumerate(training_loader):
		torch.cuda.empty_cache()

		data = data.to(device)
		label = label.to(device)
		model.zero_grad()
		output = model(data)
		loss = criterion(output, label)
		running_loss += loss.item()

		print('epoch: ', epoch, 'dataset progress: ', idx, 'loss: ', running_loss/(idx+1), end = '\r')

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		

output = model(training_dataset_raw[20].unsqueeze(0).to(device))
plt.imshow(im_convert(output.detach()))
plt.show()



