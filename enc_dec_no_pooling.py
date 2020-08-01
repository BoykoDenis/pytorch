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
from model import Enc_Dec
import random

print("hello")

lr = 0.00001

epochs = 10

flag = False

load_m = True

channels_noise = 1800

batch_size = 4

n_featers = 20

input_chanels = 3

dataset_size = 3000
dataset_move = random.randint(1, 7000)

mod_save_path = "models\\model_alpha_second.pth.tar"
mod_load_path = "models\\model_alpha_second_ijli.pth.tar"



resolutions = {"hd": [720, 1280],

			  "full-hd": [1080, 1920]}

resolution = "hd"



if resolution == "hd":

	kernel_sizes = [5, 4, 4, [3, 4], [3, 4]]

elif resolution == "full-hd":

	kernel_sizes = [5, 4, [3, 4], [3, 4], 3, 2]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fixed_noise = torch.randn(1, channels_noise, 1, 1).to(device)



def im_convert(tensor):

	image = tensor.cpu().clone().detach().numpy()

	#clone tensor --> detach it from computations --> transform to numpy

	image = image.squeeze()

	#image = image.transpose(1, 2, 0)

	# swap axis from(1,28,28) --> (28,28,1)

	#image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))

	#denormalize image

	#image = image.clip(0, 1)

	#sets image range from 0 to 1

	return image*255





def save_mod(state, filename = mod_save_path):

	torch.save(state, filename)

def load_mod(checkpoint):

	model.load_state_dict(checkpoint['state_dictionary'])
	#optimizer.load_state_dict(['state_dictionary'])


#training_path = ""
training_path = 'D:\\Datasets\\Mountain\\mountain\\'

training_dataset_raw = [Image.open(training_path+"train (" + str(i*2) + ").jpg") for i in range(1, dataset_size)]

transform_train = transforms.Compose([transforms.Resize((360, 640)),
									  transforms.ToTensor(),

									  transforms.Normalize((0.5, 0.5, 0.5),

														   (0.5, 0.5, 0.5)),

									  ])
for idx, img in enumerate(training_dataset_raw):

	training_dataset_raw[idx] = transform_train(img)

	if idx%100 == 0:

		print(idx, " images have been converted", end = "\r")



training_dataset = torch.stack(training_dataset_raw)
training_dataset_raw = None 
print(type(training_dataset))
training_dataset = torch.utils.data.TensorDataset(training_dataset, training_dataset)
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
training_dataset = None

#torch.reshape(training_dataset, (-1, batch_size))
torch.cuda.empty_cache()
#sys.exit()
	
criterion = nn.MSELoss(reduction="sum")
model = Enc_Dec(3, n_featers).to(device)
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr = lr)



if load_m:

	load_mod(torch.load(mod_load_path))

for epoch in range(epochs):
	torch.cuda.empty_cache()
	checkpoint = {"state_dictionary" : model.state_dict(), "optimizer": optimizer.state_dict()}
	running_loss = 0.0

	if epoch % 2 == 0:

		save_mod(checkpoint)

	for idx, [data, label] in enumerate(training_loader):

		torch.cuda.empty_cache()
		data = data.to(device)
		label = label.to(device)
		model.zero_grad()
		#print(type(label))
		output, _ = model(data)
		#print(output.shape)
		loss = criterion(output, label)
		running_loss += loss.item()

		print("epoch: ", epoch, "dataset progress: ", idx, "loss: ", running_loss/(idx+1), end = "\r")
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

output = model(training_dataset_raw[0].unsqueeze(0).to(device))
plt.imshow(im_convert(output.detach()))
plt.show()







