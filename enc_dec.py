import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#import pickle
from PIL import Image

lr = 0.001
epochs = 10
flag = True
load_m = False
channels_noise = 1800
batch_size = 10
mod_save_path = 'models\\model_alpha_1.pth.tar'

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
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(['state_dict'])

class Enc_Dec(nn.Module):
	def __init__(self, input_chanels, n_featers):
		super().__init__()
		
		self.conv1 = nn.Conv2d(input_chanels, n_featers*3*input_chanels, kernel_size = 5, stride = 1, padding = 2)
		self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 3, return_indices = True)
		self.conv2 = nn.Conv2d(n_featers*3*input_chanels, n_featers*6*input_chanels, kernel_size = 5, stride = 1, padding = 2)
		self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)
		self.conv3 = nn.Conv2d(n_featers*6*input_chanels, n_featers*12*input_chanels, kernel_size = 5, stride = 1, padding = 2)
		self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)
		self.conv4 = nn.Conv2d(n_featers*12*input_chanels, n_featers*24*input_chanels, kernel_size = 5, stride = 1, padding = 2)
		self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)
		self.conv5 = nn.Conv2d(n_featers*24*input_chanels, n_featers*24*5*input_chanels, kernel_size = 5, stride = 1, padding = 2)
		self.pool5 = nn.MaxPool2d(kernel_size = 5, stride = 5, return_indices = True)
		self.pool6 = nn.MaxPool2d(kernel_size = [3, 4], stride = [3, 4], return_indices = True)
		self.pool7 = nn.MaxPool2d(kernel_size = [3, 4], stride = [3, 4], return_indices = True)

		self.unpool7 = nn.MaxUnpool2d(kernel_size = [3, 4], stride = [3, 4])
		self.unpool6 = nn.MaxUnpool2d(kernel_size = [3, 4], stride = [3, 4])
		self.unpool5 = nn.MaxUnpool2d(kernel_size = 5, stride = 5)
		self.conv5_r = nn.ConvTranspose2d(n_featers*24*5*input_chanels, n_featers*24*input_chanels, kernel_size = 5, stride = 1, padding = 2)
		self.unpool4 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
		self.conv4_r = nn.ConvTranspose2d(n_featers*24*input_chanels, n_featers*12*input_chanels, kernel_size = 5, stride = 1, padding = 2)
		self.unpool3 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
		self.conv3_r = nn.ConvTranspose2d(n_featers*12*input_chanels, n_featers*6*input_chanels, kernel_size = 5, stride = 1, padding = 2)
		self.unpool2 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
		self.conv2_r = nn.ConvTranspose2d(n_featers*6*input_chanels, n_featers*3*input_chanels, kernel_size = 5, stride = 1, padding = 2)
		self.unpool1 = nn.MaxUnpool2d(kernel_size = 3, stride = 3)
		self.conv1_r = nn.ConvTranspose2d(n_featers*3*input_chanels, input_chanels, kernel_size = 5, stride = 1, padding = 2)



	def forward(self, x):
		global flag
		
		
		x = F.relu(self.conv1(x))
		x, ind1 = self.pool1(x)
		x = F.relu(self.conv2(x))
		x, ind2 = self.pool2(x)

		

		x = F.relu(self.conv3(x))
		x, ind3 = self.pool3(x)
		x = F.relu(self.conv4(x))
		x, ind4 = self.pool4(x)
		x = F.relu(self.conv5(x))
		
		x, ind5 = self.pool5(x)
		x, ind6 = self.pool6(x)
		x, ind7 = self.pool7(x)

		#print('ind00000000000000000000000000000000000000000', ind1)
		#plt.imshow(x[0, 0].cpu().detach().numpy())
		#plt.show()
		#print('Pooling... done', end = '\r')
		#print('UnPooling... start', end = '\r')
		x = self.unpool7(x, ind7)
		x = self.unpool6(x, ind6)
		x = self.unpool5(x, ind5)

		x = F.relu(self.conv5_r(x))
		x = self.unpool4(x, ind4)
		x = F.relu(self.conv4_r(x))
		x = self.unpool3(x, ind3)
		x = F.relu(self.conv3_r(x))
		
		x = self.unpool2(x, ind2)
		x = F.relu(self.conv2_r(x))
		x = self.unpool1(x, ind1)
		x = F.relu(self.conv1_r(x))

		#print('Pooling... done', end = '\r')
		
		if flag:

			indices = {'ind1': ind1, 'ind2': ind2, 'ind3': ind3, 'ind4': ind4, 'ind5': ind5, 'ind6': ind6, 'ind7': ind7}
			torch.save(indices, 'models\\indices')
			#file = open('models\\indices.txt', 'wb')
			#pickle.dump(indices, file)
			#file.close()
			flag = False
		
		return x



training_path = 'D:\\Datasets\\Mountain_railroads\\train\\'
training_dataset = [Image.open(training_path + 'mr_' + str(i) + '.jpg') for i in range(1, 64)]

transform_train = transforms.Compose([transforms.Resize((1080, 1920)),
									  transforms.ToTensor(),
									  transforms.Normalize((0.5, 0.5, 0.5),
														   (0.5, 0.5, 0.5)),
									  ])

for idx, img in enumerate(training_dataset):
	training_dataset[idx] = transform_train(img)



criterion = nn.BCEWithLogitsLoss()
model = Enc_Dec(3, 5).to(device)
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr = lr)

if load_m:
	load_mod(torch.load('models\\model_alpha_1.pth.tar'))

for epoch in range(epochs):

	checkpoint = {'state_dictionary' : model.state_dict(), 'optimizer': optimizer.state_dict()}
	if epoch % 10 == 0:
		save_mod(checkpoint)
	for idx, data in enumerate(training_dataset):
		print('epoch: ', epoch, 'dataset progress: ', idx, end = '\r')

		data = data.unsqueeze(0).to(device)
		model.zero_grad()
		output = model(data).to(device)
		loss = criterion(output, data)

		loss.backward()
		optimizer.step()
		model.zero_grad()

output = model(training_dataset[10].unsqueeze(0).to(device))
plt.imshow(im_convert(output.detach()))
plt.show()



