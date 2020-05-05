import torch

w = torch.tensor(5.0, requires_grad = True)
b = torch.tensor(3.0, requires_grad = True)

def forward(x):
	y = w*x + b
	return y