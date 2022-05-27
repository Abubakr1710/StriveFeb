#import the needed libraries

import data_handler as dh           # yes, you can import your code. Cool!
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model import NeuralNetwork


x_train, x_test, y_train, y_test = dh.load_data('C:/Users/Abubakr/Documents/GitHub/StriveFeb/Chapter 03/03. MLP Regression/data/turkish_stocks.csv')
print(y_test)


# Remember to validate your model: model.eval .........with torch.no_grad() ......model.train
