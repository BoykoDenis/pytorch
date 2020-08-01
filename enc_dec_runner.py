import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from model import Decoder, Enc_Dec

channels_noise = 1440
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
load_m = True

indices = torch.load('models\\indices')
resolution = 'hd'
n_fetures = 20
fixed_noise = torch.randn(1, n_fetures*10, 4, 4).to(device)

training_path = 'D:\\Datasets\\Mountain\\mountain\\'
img = Image.open(training_path+"train (" + str(2) + ").jpg")

transform_train = transforms.Compose([transforms.Resize((360, 640)),
									  transforms.ToTensor(),

									  transforms.Normalize((0.5, 0.5, 0.5),

														   (0.5, 0.5, 0.5)),

									  ])

img = transform_train(img)
img = img.unsqueeze(0).to(device)
def im_convert(tensor):
	image = tensor.cpu().clone().detach().numpy()
	#clone tensor --> detach it from computations --> transform to numpy
	image = image.squeeze()
	#image = image.transpose(1, 2, 0)
	print(image.shape)
	# swap axis from(1,28,28) --> (28,28,1)
	#image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
	#denormalize image
	#image = image.clip(0, 1)
	#sets image range from 0 to 1
	return image

indi = Enc_Dec(3, n_fetures).to(device)
_, indices = indi(img)
model = Decoder(3, n_fetures).to(device)
#model_orig = enc_dec.Enc_Dec(3, 5).to(device)

#print(model.state_dict())
#print(len(model_orig.state_dict()))

def load_mod(checkpoint):
	model.load_state_dict(checkpoint['state_dict'])
	#optimizer.load_state_dict(['state_dict'])
keys = ['conv5_1r.weight', 'conv5_1r.bias', 'conv5_2r.weight', 'conv5_2r.bias', 'conv5_3r.weight', 'conv5_3r.bias',
		'last_conv4.weight', 'last_conv4.bias',
		'conv4_1r.weight', 'conv4_1r.bias', 'conv4_2r.weight', 'conv4_2r.bias',
		'last_conv3.weight', 'last_conv3.bias',
		'conv3_1r.weight', 'conv3_1r.bias', 'conv3_2r.weight', 'conv3_2r.bias', 'conv3_3r.weight', 'conv3_3r.bias',
		'last_conv2.weight', 'last_conv2.bias',
		'conv2_1r.weight', 'conv2_1r.bias', 'conv2_2r.weight', 'conv2_2r.bias', 'conv2_3r.weight', 'conv2_3r.bias',
		'last_conv1.weight', 'last_conv1.bias',
		'conv1_1r.weight', 'conv1_1r.bias', 'conv1_2r.weight', 'conv1_2r.bias', 'conv1_3r.weight', 'conv1_3r.bias']

if load_m:
	model_par = torch.load("models\\model_alpha_second.pth.tar")
	model_par_load = {key: model_par.get('state_dictionary').get(key) for key in keys}
	model.load_state_dict(model_par_load)
	#print(model.state_dict())
	#print(model_par)
#print(fixed_noise)
output = model(fixed_noise, indices)
plt.imshow(im_convert(output)[1])
plt.show()

