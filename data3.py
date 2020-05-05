import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import torch.nn as nn

n_pts = 1000
X, y = datasets.make_moons(n_samples=n_pts, noise=0.1)
#datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
print(y)

X_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(1000, 1))

class Model(nn.Module):
	def __init__(self, input_size, H1, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, H1)
		self.linear2 = nn.Linear(H1, output_size)
	def forward(self, x):
		x = torch.sigmoid(self.linear(x))
		x = torch.sigmoid(self.linear2(x))
		return x
model = Model(2, 100, 1)
print(list(model.parameters()))

def predict(x):
	x[x>0.75] = -3.0
	x[x>0.50] = -2.0
	x[x>0.25] = -1.0
	x[x>0] = 0.0
	return x*-1


def scatter_plot():
	plt.scatter(X[y == 0, 0], X[y == 0, 1])
	plt.scatter(X[y == 1, 0], X[y == 1, 1])
	plt.show()

scatter_plot()


creteria = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
epoch = 1000


losses = []
for i in range(epoch):
	y_pred = model.forward(X_data)
	#print(y_pred)

	y_data=y_data
	loss = creteria(y_pred, y_data)
	print('epoch: ', i, 'loss: ', loss)
	losses.append(loss.item())


	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

plt.plot(range(epoch), losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

def plot_decision_boundary(X, y):

	x_span = np.linspace(min(X[:, 0]), max(X[:, 0]))
	y_span = np.linspace(min(X[:, 1]), max(X[:, 1]))
	xx, yy = np.meshgrid(x_span, y_span)
	grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
	pred = model.forward(grid)
	print(pred)
	#print(pred)
	z = pred.view(xx.shape).detach().numpy()
	plt.contourf(xx, yy, z)
	plt.scatter(X[y == 0, 0], X[y == 0, 1])
	plt.scatter(X[y == 1, 0], X[y == 1, 1])
	plt.show()
	

plot_decision_boundary(X, y)