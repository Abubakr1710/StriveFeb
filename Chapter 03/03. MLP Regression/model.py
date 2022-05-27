from modulefinder import Module
from turtle import forward
import torch
import torch.nn.functional as F
from torch import nn, outer
from torchsummary import summary

# YOUR CODE HERE

class NeuralNetwork(nn.Module):
    def __init__(self,input_dim, num_hidden1, num_hidden2,num_hidden3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, num_hidden1)
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear3 = nn.Linear(num_hidden2, num_hidden3)
        self.linear4 = nn.Linear(num_hidden3, 1)
        self.tanh = nn.Tanh()
        
    def forward(self,x):
        layer1 =  self.linear1(x)
        act1 = self.tanh(layer1)
        layer2 = self.linear2(act1)
        act2 = self.tanh(layer2)
        layer3 = self.linear3(act2)
        act3 = self.tanh(layer3)
        layer4 = self.linear4(act3)
        output = self.tanh(layer4)
        return output



        