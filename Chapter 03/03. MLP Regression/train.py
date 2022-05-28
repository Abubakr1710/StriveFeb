#import the needed libraries


from pickletools import optimize
from pyexpat import model
from re import X
from statistics import mode
from turtle import shape
import data_handler as dh           # yes, you can import your code. Cool!
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model import NeuralNetwork

torch.manual_seed(0)


pth='Chapter 03/03. MLP Regression/data/turkish_stocks.csv'

X_train, X_test, y_train, y_test= dh.to_batches(pth, batch_size=4)

#print(X_train)
model = NeuralNetwork(8,8,4,2)

# Remember to validate your model: model.eval .........with torch.no_grad() ......model.train

def torch_fit(X_train,X_test,y_train,y_test, model, lr, num_epochs):
    
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(),lr)
    print_every = 1

    train_lossses = []
    test_losses=[]
    for epoch in range(num_epochs):
        running_loss=0

        epoch_list = []
        print(f"Epoch: {epoch+1}/{num_epochs}")
        for i, (X_train_batches, y_train_batches) in enumerate(zip(X_train, y_train)):
            optimizer.zero_grad()
            output = model.forward(X_train_batches)
            loss = criterion(output,y_train_batches)
            epoch_list.append(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_every == 0:
                print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
                running_loss = 0

        model.eval()
        with torch.no_grad():
            test_epoch_list=[]
            for j, (X_test_batches,y_test_batches) in enumerate(zip(X_test,y_test)):
                test_output=model.forward(X_test_batches)
                test_loss=criterion(test_output,y_test_batches)
                test_epoch_list.append(test_loss.item())

        mean_epoch_losses_test=sum(test_epoch_list)/len(test_epoch_list)
        test_losses.append(mean_epoch_losses_test)
        mean_epoch_losses=sum(epoch_list)/len(epoch_list)
        train_lossses.append(mean_epoch_losses)
        print(f'Mean epoch loss for train{mean_epoch_losses}')
        print(f'Mean epoch loss for test{mean_epoch_losses_test}')
    

    plt.plot(train_lossses,label = 'Train Losses')
    plt.plot(test_losses,label = 'Test Losses')
    plt.legend()
    plt.show()
model = torch_fit(X_train=X_train,X_test=X_test, y_train=y_train,y_test=y_test,lr=0.002,num_epochs=30, model=model)
print('X_train',X_train.shape)
print('y_train',y_train.shape)
print('X_Test', X_test.shape)
print('y_test',y_test.shape)


