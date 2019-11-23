import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import random

class SimpleANN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(SimpleANN, self).__init__()
        self.input = nn.Linear(input_size, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden_state = self.hidden1(F.relu(self.input(x)))
        return self.output(F.relu(hidden_state)).view(batch_size, -1)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop_prob=0.0):
        super(SimpleLSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x, hidden):
        batch_size = x.size(1)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(self.dropout(hidden[0].view(batch_size, self.hidden_dim)))
        return out, hidden
    
    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).double(),
                      torch.zeros(self.n_layers, batch_size, self.hidden_dim).double())
        return hidden

class TimeseriesSampler(Sampler):
    """Samples sequences from the dataset using the given window size and step size, while accounting for
    	gaps in the time_index

    	time_index (numpy.Array)
    """
    def __init__(self, time_index, window_size=5, step_size=1, shuffle=False):
        self.time_index = time_index
        self.windows = []
        i = 0
        n = len(time_index)
        left_bound = 0
        while i < n:
        	if i-left_bound+1 < window_size:
        		pass
        	elif time_index[i] != time_index[left_bound]+window_size-1:
        		left_bound = i
        	else:
        		self.windows.append(list(range(left_bound, i+1)))
        		left_bound += 1
        	i+=1
        if shuffle:
           random.shuffle(self.windows)
        
    def __iter__(self):
        return iter(self.windows)

    def __len__(self):
        return len(self.windows)