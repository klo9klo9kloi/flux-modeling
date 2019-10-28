import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(SimpleAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h_out = self.fc1(x) 
        out = self.fc2(h_out)
        out = out.view(batch_size, -1)
        return out, h_out.view(batch_size, -1)