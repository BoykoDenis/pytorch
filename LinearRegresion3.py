import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

X = torch.randn(100, 1)*10
#print(X)
Y = X + torch.randn(100, 1)

#plt.plot(X.numpy(), Y.numpy(), 'o')
#plt.show()

class LR(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, output_size)
	def forward(self, x):
		pred = self.linear(x)
		return pred


torch.manual_seed(1)

model = LR(1, 1)

print(list(model.parameters()))

[w, b] = model.parameters()
print(w, b)
w1 = w[0][0]
b1 = b[0]

print(w1, b1)

def get_params():
	return w[0][0].item(), b[0].item()

def plot_fit(title):
	plt.title = title
	w1, b1 = get_params()
	x1 = np.array([-30, 30])
	print(x1)
	y1 = w1*x1+b1
	plt.plot(x1, y1, 'r')
	plt.scatter(X, Y)
	plt.show()

plot_fit('initial_model')
