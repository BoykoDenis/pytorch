import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

channels_noise = 1800
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
load_m = True
fixed_noise = torch.randn(1, channels_noise, 1, 1).to(device)
indices = torch.load('models\\indices')


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

class Decoder(nn.Module):
	def __init__(self, input_chanels, n_featers):
		super().__init__()

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

	def forward(self, x, ind):
		x = self.unpool7(x, ind['ind7'])
		x = self.unpool6(x, ind['ind6'])
		x = self.unpool5(x, ind['ind5'])
		x = F.relu(self.conv5_r(x))
		x = self.unpool4(x, ind['ind4'])
		x = F.relu(self.conv4_r(x))
		x = self.unpool3(x, ind['ind3'])
		x = F.relu(self.conv3_r(x))
		x = self.unpool2(x, ind['ind2'])
		x = F.relu(self.conv2_r(x))
		x = self.unpool1(x, ind['ind1'])
		x = F.relu(self.conv1_r(x))
		return x
model = Decoder(3, 5).to(device)
#model_orig = enc_dec.Enc_Dec(3, 5).to(device)

#print(model.state_dict())
#print(len(model_orig.state_dict()))

def load_mod(checkpoint):
	model.load_state_dict(checkpoint['state_dict'])
	#optimizer.load_state_dict(['state_dict'])
keys = ['conv5_r.weight', 'conv5_r.bias', 'conv4_r.weight', 'conv4_r.bias', 'conv3_r.weight', 'conv3_r.bias', 'conv2_r.weight', 'conv2_r.bias', 'conv1_r.weight', 'conv1_r.bias']
if load_m:
	model_par = torch.load('models\\model_alpha_1.pth.tar')
	model_par_load = {key: model_par.get('state_dictionary').get(key) for key in keys}
	model.load_state_dict(model_par_load)
	#print(model.state_dict())
	#print(model_par)
#print(fixed_noise)
output = model(fixed_noise, indices)
plt.imshow(im_convert(output))
plt.show()

