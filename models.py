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


class CGAN_Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim, hidden_dim_base):
        super(CGAN_Generator, self).__init__()

        # self.noise_dim = noise_dim
        # self.label_dim = label_dim
        self.output_dim = output_dim

        def block(in_dim, out_dim, normalize=True):
            layers = [nn.Linear(in_dim, out_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.noise_block = nn.Sequential(
            *block(noise_dim, hidden_dim_base, normalize=False)
            )
        self.label_block = nn.Sequential(
            *block(label_dim, hidden_dim_base, normalize=False)
            )

        self.sequence = nn.Sequential(
            *block(2*hidden_dim_base, 2*hidden_dim_base),
            *block(2*hidden_dim_base, 4*hidden_dim_base),
            *block(4*hidden_dim_base, 8*hidden_dim_base),
            nn.Linear(8*hidden_dim_base, output_dim)
            #nn.Tanh()
        )

    def forward(self, z, y):
        z = self.noise_block(z)
        y = self.label_block(y)

        out = torch.cat([z, y], dim=1)
        out = self.sequence(out)
        # out = out.view(out.size(0), self.output_dim)
        return out

class CGAN_Discriminator(nn.Module):
    def __init__(self, sample_dim, label_dim, hidden_dim_base):
        super(CGAN_Discriminator, self).__init__()

        # self.sample_dim = sample_dim
        # self.label_dim = label_dim

        def block(in_dim, out_dim):
            layers = [nn.Linear(in_dim, out_dim)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.sample_block = nn.Sequential(
            *block(sample_dim, 2*hidden_dim_base)
            )
        self.label_block = nn.Sequential( 
            *block(label_dim, 2*hidden_dim_base)
            )

        self.sequence = nn.Sequential(
            nn.Linear(4*hidden_dim_base, 2*hidden_dim_base),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2*hidden_dim_base, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        x = self.sample_block(x)
        y = self.label_block(y)
        out = torch.cat([x, y], dim=1)
        validity = self.sequence(out)
        return validity
