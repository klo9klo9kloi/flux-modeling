import lstm_model as lstm
import numpy as np 
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator

class SimpleLSTMRegressor(BaseEstimator):
	# seq len corresponds to how big we want the 'memory window' of our model to be
	def __init__(self, lr=0.05, batch_size=1, seq_len = 5, epochs = 5, clip = 5, input_dim = 5, output_dim=1, hidden_dim=512, n_layers=3):
		self.lr = lr
		self.batch_size = batch_size
		self.sequence_length = seq_len
		self.epochs = epochs
		self.clip = clip
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.model = lstm.SimpleLSTM(input_dim, output_dim, hidden_dim, n_layers)
	
	def get_params(self, deep=False):
		return {"lr": self.lr, "batch_size": self.batch_size, "seq_len": self.sequence_length, "epochs": self.epochs, "clip": self.clip, "input_dim": self.input_dim, "output_dim": self.output_dim, "hidden_dim": self.hidden_dim, "n_layers": self.n_layers}

	def set_params(self, **params):
		self.lr = params['lr']
		self.batch_size = params['batch_size']
		self.sequence_length = params['seq_len']
		self.epochs = params['epochs']
		self.clip = params['clip']
		self.input_dim = params['input_dim']
		self.output_dim = params['output_dim']
		self.hidden_dim = params['hidden_dim']
		self.n_layers = params['n_layers']
		self.model = lstm.SimpleLSTM(params['input_dim'], params['output_dim'], params['hidden_dim'], params['n_layers'])
		return self

	def fit(self, X, y, **kwargs):
		data = TensorDataset(torch.from_numpy(X[:, :-1]), torch.from_numpy(y))
		# we use seq_len+1 here so that when we generate windows, we can access the index of the day we want to predict
		loader = DataLoader(data, shuffle=False, batch_size=self.batch_size, sampler=lstm.TimeseriesSampler(X[:, -1:].squeeze().astype('int'), self.sequence_length+1, shuffle=True))
		
		is_cuda = torch.cuda.is_available()
		if is_cuda:
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")
		self.model.to(device)

		#set up loss function and optimizer
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		
		self.model.train()
		# train
		for i in range(self.epochs):
			h = self.model.init_hidden(self.batch_size)
			for inpts, lbls in loader:
				h = tuple([e.to(device).data for e in h])
				inpts, lbls = inpts.to(device), lbls.squeeze().to(device)
				self.model.zero_grad()
				output, h = self.model(inpts, h)

				# we only care about the last output for now
				loss = criterion(output.squeeze()[-1], lbls.float()[-1])
				loss.backward()
				nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
				optimizer.step()
				print("Epoch: {}/{}...".format(i+1, self.epochs), "Loss: {:.6f}...".format(loss.item()))
		return self

	def predict(self, X, **kwargs):
		data = TensorDataset(torch.from_numpy(X[:, :-1]))
		loader = DataLoader(data, shuffle=False, batch_size=self.batch_size, sampler=lstm.TimeseriesSampler(X[:, -1:].squeeze().astype('int'), self.sequence_length+1, shuffle=True))
		
		is_cuda = torch.cuda.is_available()
		if is_cuda:
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")

		self.model.eval()
		h = self.model.init_hidden(self.batch_size)
		predictions = []

		for inp in loader:
			h = tuple([e.to(device).data for e in h])
			inp = inp[0].to(device)
			out, h = self.model(inp, h)
			predictions.append(out.squeeze()[-1].item())
		return predictions


	def score(self, X, y, **kwargs):
		predictions = self.predict(X, **kwargs)
		truth = []
		i = 0
		time_index = X[:, -1:].squeeze().astype('int')
		print(time_index)
		n = len(time_index)
		left_bound = 0
		while i < n:
			if i-left_bound == self.sequence_length:
				if time_index[left_bound] == time_index[i] - self.sequence_length:
					truth.append(y[i])
					left_bound += 1
				else:
					left_bound = i
			i+=1
		return r2_score(truth, predictions)






