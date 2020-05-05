import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import requests
import PIL
from torchvision import datasets, transforms 
from torch import nn
from PIL import Image


transform = transforms.Compose([transforms.Resize((28, 28)),
								transforms.ToTensor(),
								transforms.Normalize((0.5,), (0.5,))])
#compose 2 methods into one
# - convert image from range 0-255 into 0-1
# - Normalize data(for each color chenel) (I) - mean; (II) - standart deviation; ([max-min img value]-mean)/std
#   - convert image range from [-1] -- [1]

training_dataset = datasets.MNIST(root = './data', train = True, download = False, transform =transform)
validation_dataset = datasets.MNIST(root = './data', train = False, download = False, transform =transform)
# traingng dataset

training_loader = torch.utils.data.DataLoader(dataset = training_dataset, batch_size = 100, shuffle = True)
validation_loader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = 100, shuffle = False)
#training set up (batch size)
print('testpoint 1')


def im_convert(tensor):
	print('converting image...')
	image = tensor.clone().detach().numpy()
	#clone tensor --> detach it from computations --> transform to numpy
	image = image.transpose(1, 2, 0)
	# swap axis from(1,28,28) --> (28,28,1)
	image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
	#denormalize image
	image = image.clip(0, 1)
	#sets image range from 0 to 1
	return image

print('data iterating...')
dataiter = iter(training_loader)
print('pushing data...')
images, labels = dataiter.next()
print('initializing new vars...')
fig = plt.figure(figsize = (25, 4))
print('shaping data...')

for idx in np.arange(20):
	ax = fig.add_subplot(2, 10, idx+1)
	#plt.imshow(im_convert(images[idx]))
	ax.set_title(labels[idx].item())



class LeNet(nn.Module):
	def __init__(self, D_in, H1, H2, D_out):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		# input chanell, 
	def forward(self, x):
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x) # no act func (crossentropyloss)
		return x


model = Classifier(784, 125, 65, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 12
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_hisotory = []

for e in range(epochs):
	running_loss = 0.0
	#keeps tracking of the loss due every epoch

	running_corrects = 0.0
	#accuracy calculating

	val_running_loss = 0.0
	val_running_corrects = 0.0

	for inputs, labels in training_loader:
		
		inputs = inputs.view(inputs.shape[0], -1)
		#converts tensor to array with shape of inputs x axis, with -1 as automatic calculation of y

		outputs = model(inputs)
		# pushes the input of nn into forward function threw nn and save it to output

		loss = criterion(outputs, labels)
		#calculating the error(loss) based on multi class nn CategoricalCrossEntropy loss function

		optimizer.zero_grad()
		#reset the gradient to zero from previous step, in order to prevent using the same value of slope
		loss.backward()
		#calculate derivatives for back propagation
		optimizer.step()
		#optimizing weights in direction of slope

		_, preds = torch.max(outputs, 1)
		#takes the index of maximal value from the output

		running_corrects += torch.sum(preds == labels.data)

		running_loss += loss.item()
	else:
		with torch.no_grad():
			for val_inputs, val_labels in validation_loader:
				val_inputs = val_inputs.view(val_inputs.shape[0], -1)
				val_outputs = model(val_inputs)
				val_loss = criterion(val_outputs, val_labels)
				_, val_preds = torch.max(val_outputs, 1)
				val_running_loss += val_loss.item()
				val_running_corrects += torch.sum(val_preds == val_labels.data) 
			epoch_loss = running_loss/len(training_loader)
			#calculates the avarage loss due the epoch
			epoch_acc = running_corrects.float()/len(training_loader)

			val_epoch_loss = val_running_loss/len(training_loader)
			val_epoch_acc = val_running_corrects.float()/len(validation_loader)
			running_loss_history.append(epoch_loss)
			# add epoch loss to the history

			running_corrects_history.append(epoch_acc)

			val_running_loss_history.append(val_epoch_loss)
			val_running_corrects_hisotory.append(val_epoch_acc)

			print('training loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))
			print('validation loss: {:.4f}, acc: {:.4f}'.format(val_epoch_loss, val_epoch_acc))
plt.plot(running_loss_history)
plt.plot(running_corrects_history)
plt.show() 

url = 'https://images.homedepot-static.com/productImages/007164ea-d47e-4f66-8d8c-fd9f621984a2/svn/architectural-mailboxes-house-letters-numbers-3585b-5-64_1000.jpg'
response = requests.get(url, stream = True)
img = Image.open(response.raw)
plt.imshow(img)

img = PIL.ImageOps.invert(img)
plt.imshow(img)
# invert image colors white and black
img = img.convert('1')
plt.imshow(img)
#converts the image into binary black and white
img = transform(img)

plt.imshow(im_convert(img))
print(response)
plt.show()
print(img.shape)
img = img.view(img.shape[0], -1)
outputs = model(img)
_, pred = torch.max(outputs, 1)
print(pred.item())