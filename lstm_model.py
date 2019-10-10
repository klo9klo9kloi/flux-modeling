import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
import random

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop_prob=0.5):
        super(SimpleLSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        out = out.view(batch_size, -1)
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
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
        # print(time_index)
        # print(self.windows)
        # mappings = []
        # for window in self.windows:
        # 	mapping = []
        # 	for index in window:
        # 		mapping.append(time_index[index])
        # 	mappings.append(mapping)
        # print(mappings)
        if shuffle:
            random.shuffle(self.windows)

        
    def __iter__(self):
        return iter(self.windows)

    def __len__(self):
        return len(self.windows)