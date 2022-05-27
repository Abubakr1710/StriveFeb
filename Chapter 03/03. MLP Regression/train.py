#import the needed libraries

from pickletools import optimize
from pyexpat import model
from turtle import shape
import data_handler as dh           # yes, you can import your code. Cool!
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model import NeuralNetwork

pth='Chapter 03/03. MLP Regression/data/turkish_stocks.csv'

X_train, X_test, y_train, y_test= dh.to_batches(pth, batch_size=100)

#print(X_train)
model = NeuralNetwork(X_train.shape[0], 32,16,8)

# Remember to validate your model: model.eval .........with torch.no_grad() ......model.train

def torch_fit(X_train,y_train, model, loss, lr, num_epochs, batches ):
    
    criterion = nn.L1Loss(model.parameters)
    optimizer = optim.SGD()


    for epoch in range(num_epochs):


        print(f"Epoch: {epoch+1}/{num_epochs}")
        for i in range(X_train.shape[0]):
            optimizer.zero_grad()
            output = model.forward(X_train)
            loss = criterion(output,y_train)
            


