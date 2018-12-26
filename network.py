import torch
from torch import nn
import torch.nn.functional as F

#The class that defined the classifier
class Classifier(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,drop_p=0.2):
        super().__init__()
        #the first layer
        self.hidden_layer = nn.ModuleList([nn.Linear(input_size,hidden_size[0])])
        layer_structure = zip(hidden_size[:-1],hidden_size[1:])
        #rest of the hidden layer
        self.hidden_layer.extend([nn.Linear(h1,h2) for h1,h2 in layer_structure])
        #output layer
        self.output_layer = nn.Linear(hidden_size[-1],output_size)
        #dropout
        self.dropout = nn.Dropout(drop_p)
        
    def forward(self,x):
        for linear in self.hidden_layer:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return F.log_softmax(x,dim=1)       