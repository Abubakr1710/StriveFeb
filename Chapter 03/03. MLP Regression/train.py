#import the needed libraries

from pickletools import optimize
from pyexpat import model
from statistics import mode
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
model = NeuralNetwork(8,16,8,4)

# Remember to validate your model: model.eval .........with torch.no_grad() ......model.train

def torch_fit(X_train,y_train, model, lr, num_epochs):
    
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(),lr)

    train_lossses = []
    for epoch in range(num_epochs):

        epoch_list = []
        print(f"Epoch: {epoch+1}/{num_epochs}")
        for i in range(X_train.shape[0]):
            optimizer.zero_grad()
            output = model.forward(X_train)
            loss = criterion(output,y_train)
            epoch_list.append(loss.item())
            loss.backward()
            optimizer.step()

        mean_epoch_losses=sum(epoch_list)/len(epoch_list)
        train_lossses.append(mean_epoch_losses)
    

    plt.plot(train_lossses)
    plt.show()
model = torch_fit(X_train=X_train, y_train=y_train,lr=0.003,num_epochs=20, model=model)


