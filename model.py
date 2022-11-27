import torch
import torch.nn as nn
        
class tdActor(nn.Module):
    def __init__(self, states = 24, actions = 2, layer1 = 128, layer2=128):
        super().__init__()
        self.L1 = nn.Linear(states, layer1)
        self.L2 = nn.Linear(layer1, layer2)
        self.out = nn.Linear(layer2, actions) 
 
        self.activation = nn.ReLU()
         
    ## implementation of forward pass 
    def forward(self, states):
        out =  self.activation(self.L1(states))
        out =  self.activation(self.L2(out))
        out = torch.tanh(self.out(out))
 
        return out 
    
class Critic(nn.Module):
    def __init__(self, states = 48, actions = 4, layer1 = 128, layer2 = 128):
        super().__init__()
        
        self.L1 = nn.Linear(states+actions, layer1)
        self.L2 = nn.Linear(layer1, layer2)
        self.out1 = nn.Linear(layer2, 1)
        
        self.L3 = nn.Linear(states+actions, layer1)
        self.L4 = nn.Linear(layer1, layer2)
        self.out2 = nn.Linear(layer2, 1)
        
        self.activation = nn.ReLU()
        
    def forward(self, states, actions):
#         print(states.size(), actions.size())
        inputs = torch.cat([states, actions], 1)
    
        q1 =  self.activation(self.L1(inputs))
        q1 =  self.activation(self.L2(q1))
        q1 = self.out1(q1)
        
        q2 =  self.activation(self.L3(inputs))
        q2 =  self.activation(self.L4(q2))
        q2 = self.out2(q2)       
        
        return q1, q2
    
    def Q1(self, states, actions):
        inputs = torch.cat([states, actions], 1)
        q1 =  self.activation(self.L1(inputs))
        q1 =  self.activation(self.L2(q1))
        q1 = self.out1(q1)     
        return q1