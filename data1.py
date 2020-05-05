import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import torch.nn as nn

n_pts = 100
centers = [[-0.5, 0.5], [0.5, -0.5]]
X, y = datasets.make_blobs(n_samples = n_pts, centers = centers, cluster_std = 0.4)

X_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(100, 1))

class Model(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, output_size)
	def forward(self, x):
		pred = torch.sigmoid(self.linear(x))
		return pred
model = Model(2, 1)
print(list(model.parameters()))
[w, b] = model.parameters()
w1, w2 = w.view(2)
def get_param():
 return w1.item(), w2.item(), b[0].item()

def scatter_plot():
	plt.scatter(X[y == 0, 0], X[y == 0, 1])
	plt.scatter(X[y == 1, 0], X[y == 1, 1])
	plt.show()

def plot_fit(title):
	w1, w2, b1 = get_param()
	plt.title = title
	x1 = np.array([-2.0, 2.0])
	x2 = (x1*w1+b1)/-w2
	print(x2)
	plt.plot(x1, x2, 'r')
	scatter_plot()


creteria = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
epoch = 1000


losses = []
for i in range(epoch):
	y_pred = model.forward(X_data)
	loss = creteria(y_pred, y_data)
	print('epoch: ', i, 'loss: ', loss)
	losses.append(loss.item())


	optimizer.zero_grad()
	loss.backward()
	optimizer.step()



plot_fit('Model_fit')
