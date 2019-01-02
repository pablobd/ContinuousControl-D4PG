import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    " Actor (Policy) Model - Neural net to decide what action the agent must take "
    
    def __init__(self, action_size, state_size, seed, hidden_layers, init_weights):
        """
        Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers (int list): Number of hidden layers and nodes in each hidden layer
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        # initial layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # final layer
        self.output = nn.Linear(hidden_layers[-1], action_size)
        
        # send networks to device
        for linear in self.hidden_layers:
            linear.to(device)
        
        self.output.to(device)
        
        if init_weights:
            print("initializing weights...")
            self.initialize_weights()
        
    def initialize_weights(self):
        " Initialize weights of layers "
        
        for layer in self.hidden_layers:
            layer.weight.data.uniform_(*hidden_init(layer))
            
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        
        
    def forward(self, x):
        """ 
        Forward network that maps state -> action 
        
        Params
        ======
            x (tensor): State vector
        """
        
        # forward through each layer in `hidden_layers`, with ReLU activation
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
        
        # forward final layer with tanh activation (-1, 1)
        return torch.tanh(self.output(x))
        
class Critic(nn.Module):
    " Critic (Value) Model - Neural net to estimate the total expected episodic return associated to one action in a given state "
    
    def __init__(self, action_size, state_size, seed, hidden_layers, init_weights):
        """
        Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers (int list): Number of hidden layers and nodes in each hidden layer
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        # initial layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # hidden layers
        hidden_layers[0] += action_size
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # final layer
        self.output = nn.Linear(hidden_layers[-1], 1)
        
        # send networks to device
        for linear in self.hidden_layers:
            linear.to(device)
        
        self.output.to(device)
        
        if init_weights:
            print("initializing weights...")
            self.initialize_weights()
        
    def initialize_weights(self):
        " Initialize weights of layers "
        
        for layer in self.hidden_layers:
            layer.weight.data.uniform_(*hidden_init(layer))
            
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, states, actions):
        """ 
        Forward network that maps state -> action 
        
        Params
        ======
            states (tensor): State vector
            actions (tensor): Action vector
        """
        
        # forward through first layer
        x = F.leaky_relu(self.hidden_layers[0](states))
        
        # concatenate output of first layer and action vector
        x = torch.cat((x, actions), dim = 1)
        
        # forward through each layer in `hidden_layers`, with Leaky ReLU activation
        for linear in self.hidden_layers[1:]:
            x = F.leaky_relu(linear(x))
        
        # forward final layer with tanh activation (-1, 1)
        return self.output(x)
    