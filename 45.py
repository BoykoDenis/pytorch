import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

training_dataset = datasets.MNIST(root = './data', train = True, download = True)