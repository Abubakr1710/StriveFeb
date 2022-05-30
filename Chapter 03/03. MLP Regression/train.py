#import the needed libraries


from cProfile import label
import imp
from pickletools import optimize
from pyexpat import model
from re import X
from statistics import mode
from turtle import shape
import data_handler as dh           # yes, you can import your code. Cool!
import torch
import torch.optim as optim
import torch.nn as nn
#import ignite.contrib.metrics.regression.R2Score as
from ignite.contrib.metrics.regression import r2_score
import matplotlib.pyplot as plt
import numpy as np

from model import NeuralNetwork

torch.manual_seed(0)


pth='Chapter 03/03. MLP Regression/data/turkish_stocks.csv'

X_train, X_test, y_train, y_test= dh.to_batches(pth, batch_size=16)

#print(X_train)
model = NeuralNetwork(8,200,100,50)

# Remember to validate your model: model.eval .........with torch.no_grad() ......model.train




def torch_fit(X_train,X_test,y_train,y_test, model, lr, num_epochs):
    
   
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(),lr)
    print_every = 1

    train_lossses = []
    test_losses=[]
    acc=[]
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
                #print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
                running_loss = 0

        model.eval()
        with torch.no_grad():
            test_epoch_list=[]
            mape_list=[]
            for j, (X_test_batches,y_test_batches) in enumerate(zip(X_test,y_test)):
                test_output=model.forward(X_test_batches)
                test_loss=criterion(test_output,y_test_batches)
                #test_loss_r2=criterion(y_test_batches, test_output)
                target_mean=torch.mean(y_test_batches)
                ss_tot=torch.sum((y_test_batches - target_mean)**2)
                ss_res=torch.sum((y_test_batches - test_loss)**2)
                r2=torch.abs(1 - ss_res/ss_tot)
                test_epoch_list.append(test_loss.item())
                #mape=torch.abs((y_test_batches[0]-X_test_batches[0]) / y_test_batches[0])*100
                
                mape_list.append(r2)



                #mape = np.abs((actual - predicted) / actual).mean(axis=0) * 100

        mean_epoch_losses_test=sum(test_epoch_list)/len(test_epoch_list)
        test_losses.append(mean_epoch_losses_test)
        mean_epoch_losses=sum(epoch_list)/len(epoch_list)
        train_lossses.append(mean_epoch_losses)
        mean_mape = sum(mape_list)/len(mape_list)
        acc.append(mean_mape)


        print(f'Mean epoch loss for train: {mean_epoch_losses}')
        print(f'Mean epoch loss for test: {mean_epoch_losses_test}')
        print(f'Mean accuracy for epoch: {mean_mape}')


        model.train()

    print(acc)

        
    
    x_axis_acc=list(range(num_epochs))
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_lossses,label = 'Train Losses')
    plt.plot(test_losses,label = 'Test Losses')

    plt.subplot(1,2,2)
    plt.plot(x_axis_acc, acc, label='R2score')
    plt.legend()
    plt.show()
    #plt.plot(acc, label='Accuracy')
    #plt.show()
model = torch_fit(X_train=X_train,X_test=X_test, y_train=y_train,y_test=y_test,lr=0.001,num_epochs=100, model=model)
print('X_train',X_train.shape)
print('y_train',y_train.shape)
print('X_Test', X_test.shape)
print('y_test',y_test.shape)