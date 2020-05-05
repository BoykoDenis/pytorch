import torch

from torch.nn import Linear

torch.manual_seed(1) #seed for random

model = Linear(in_features = 1, out_features = 1) #nomber of inputs and outputs params
#print(model.bias, model.weight)

x = torch.tensor([[2.0], [3.3]])

print(model.forward(x))