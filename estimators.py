import lstm_model as lstm
import numpy as np 
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator

def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))

class SimpleLSTMRegressor(BaseEstimator):
	# seq len corresponds to how big we want the 'memory window' of our model to be
	def __init__(self, input_dim, output_dim, hidden_dim=512, n_layers=1, lr=0.05, batch_size=1, seq_len = 5, epochs = 50, threshold=1e-5, clip = 5, scoring='mse'):
		self.lr = lr
		self.batch_size = batch_size
		self.sequence_length = seq_len
		self.epochs = epochs
		self.clip = clip
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.scoring = scoring
		self.threshold = threshold

	def stop_condition(self, iteration, loss_diff):
		if loss_diff <= self.threshold:
			return True
		return False
	
	def get_params(self, deep=False):
		return {"lr": self.lr, "batch_size": self.batch_size, 
				"seq_len": self.sequence_length, "epochs": self.epochs, "clip": self.clip, 
				"input_dim": self.input_dim, "output_dim": self.output_dim, "hidden_dim": self.hidden_dim, 
				"n_layers": self.n_layers, "scoring": self.scoring, "threshold": self.threshold}

	def set_params(self, **params):
		self.lr = params.get('lr', self.lr)
		self.batch_size = params.get('batch_size', self.batch_size)
		self.sequence_length = params.get('seq_len', self.sequence_length)
		self.epochs = params.get('epochs', self.epochs)
		self.clip = params.get('clip', self.clip)
		self.input_dim = params.get('input_dim', self.input_dim)
		self.output_dim = params.get('output_dim', self.output_dim)
		self.hidden_dim = params.get('hidden_dim', self.hidden_dim)
		self.n_layers = params.get('n_layers', self.n_layers)
		self.scoring = params.get('scoring', self.scoring)
		self.threshold = params.get('threshold', self.threshold)
		return self

	def fit(self, X, y, **kwargs):
		self.model = lstm.SimpleLSTM(self.input_dim, self.output_dim, self.hidden_dim, self.n_layers)
		# the last column of X is the time_index column, which we dont want to train on
		all_training_data = torch.from_numpy(X[:, :-1].astype('float64'))
		all_training_labels = torch.from_numpy(y.astype('float64'))
		data = TensorDataset(all_training_data, all_training_labels)
		loader = DataLoader(data, shuffle=False, batch_size=self.batch_size, sampler=lstm.TimeseriesSampler(X[:, -1:].squeeze().astype('int'), self.sequence_length))
		
		is_cuda = torch.cuda.is_available()
		if is_cuda:
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")
		self.model.double()
		self.model.to(device)

		#set up loss function and optimizer
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		
		# train
		prev_loss = np.inf
		for i in range(self.epochs):
			#train on one pass of data
			self.model.train()
			for inpts, lbls in loader:
				# input should be 3d tensor of shape (batch_size, seq_len, input_dim)
				optimizer.zero_grad()
				h = self.model.init_hidden(inpts.size(0))
				h = tuple([e.to(device).data for e in h])
				inpts, lbls = inpts.to(device), lbls.to(device)
				self.model.zero_grad()
				output, h, _ = self.model(inpts, h)
				loss = criterion(output.view(inpts.size(0), -1), lbls.view(inpts.size(0), -1))
				loss.backward()
				nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
				optimizer.step()
			print("Epoch: {}/{}...".format(i+1, self.epochs))

			# #evaluate total training loss and check stopping condition
			# self.model.eval()
			# total_loss = 0
			# with torch.no_grad():		
			# 	for inpts, lbls in loader:
			# 		# input should be 3d tensor of shape (batch_size, seq_len, input_dim)
			# 		h = self.model.init_hidden(inpts.size(0))
			# 		h = tuple([e.to(device).data for e in h])
			# 		inpts, lbls = inpts.to(device), lbls.to(device)
			# 		output, h, _ = self.model(inpts, h)

			# 		total_loss += criterion(output.view(inpts.size(0), -1)[-1], lbls.view(inpts.size(0), -1)[-1])
			# 	total_loss = total_loss/len(loader)
			# 	print("Epoch: {}/{}...".format(i+1, self.epochs), "Loss: {:.6f}...".format(total_loss.item()))
			# if self.stop_condition(i, np.abs(total_loss-prev_loss)):
			# 	break
			# prev_loss = total_loss
		self.trained_for = i
		return self

	def predict(self, X, **kwargs):
		data = TensorDataset(torch.from_numpy(X[:, :-1].astype('float64')))
		loader = DataLoader(data, shuffle=False, batch_size=self.batch_size, sampler=lstm.TimeseriesSampler(X[:, -1:].squeeze().astype('int'), self.sequence_length))
		
		is_cuda = torch.cuda.is_available()
		if is_cuda:
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")
		self.model.double()
		self.model.to(device)
		self.model.eval()
		
		predictions = []
		for inp in loader:
			inp = inp[0]
			h = self.model.init_hidden(inp.size(0))
			h = tuple([e.to(device).data for e in h])
			inp = inp.to(device)
			out, h, _ = self.model(inp, h)
			out = out.view(inp.size(0), -1)
			if out.size(0) == 1:
				predictions.append(out.squeeze()[-1].item())
			else:
				predictions += out[:, -1].squeeze().tolist()
		return predictions

	def score(self, X, y, **kwargs):
		time_index = X[:, -1:].squeeze().astype('int')
		n = len(time_index)
		if self.sequence_length >= n:
			raise ValueError("Sequence length is greater than size of test data -> cannot evaluate on test data")
		predictions = np.nan_to_num(np.array(self.predict(X, **kwargs), dtype='float64'))
		truth = np.nan_to_num(y[self.sequence_length-1:])
		if self.scoring == 'r2':
			return r2_score(truth, predictions)
		elif self.scoring == 'mda':
			return mda(truth, predictions)
		elif self.scoring == 'l1':
			return -mean_absolute_error(truth, predictions)
		else:
			return -mean_squared_error(truth, predictions)

	def r2_score(self, X, y, **kwargs):
		time_index = X[:, -1:].squeeze().astype('int')
		n = len(time_index)
		if self.sequence_length >= n:
			raise ValueError("Sequence length is greater than size of test data -> cannot evaluate on test data")
		predictions = np.nan_to_num(np.array(self.predict(X, **kwargs), dtype='float64'))
		truth = np.nan_to_num(y[self.sequence_length-1:])
		return r2_score(truth, predictions)

	def get_cell_state_data(self, X):
		data = TensorDataset(torch.from_numpy(X[:, :-1].astype('float64')))
		loader = DataLoader(data, shuffle=False, batch_size=self.batch_size, sampler=lstm.TimeseriesSampler(X[:, -1:].squeeze().astype('int'), self.sequence_length))
		
		is_cuda = torch.cuda.is_available()
		if is_cuda:
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")

		self.model.double()
		self.model.to(device)
		self.model.eval()
		cell_state_data = []

		for inp in loader:
			inp = inp[0]
			h = self.model.init_hidden(inp.size(0))
			h = tuple([e.to(device).data for e in h])

			inp = inp.to(device)
			out, h, all_cell_state = self.model(inp, h)
			cell_state_data.append(all_cell_state.contiguous().view(-1, self.hidden_dim))
		return cell_state_data






