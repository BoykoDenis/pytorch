import torch as T
from torchviz import *

x=T.ones(10, requires_grad=True)

y=x**2
z=x**3
r=(y+z).sum()
print(r)
make_dot(r)

r.backward()
print(r)
#print(x.grad)
