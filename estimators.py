from models import SimpleLSTM, SimpleANN, TimeseriesSampler
import numpy as np 
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator

def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))


class SimpleRegressorBase(BaseEstimator):
	def __init__(self, input_dim, output_dim, hidden_dim=16, lr=0.05, batch_size=1, epochs=20, threshold=1e-5, regularization_param = 0, scoring='mse'):
		self.lr = lr
		self.batch_size = batch_size
		self.epochs = epochs
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.scoring = scoring
		self.threshold = threshold
		self.regularization_param = regularization_param
		self.plateau = 0

	def stop_condition(self, iteration, curr_loss, prev_loss):
		if np.abs(curr_loss-prev_loss) <= self.threshold:
			self.plateau += 1
		else:
			self.plateau = 0
		if curr_loss > self.min_loss:
			self.tol += 1
		else:
			self.tol = 0
			self.min_loss = curr_loss
		if self.tol > 20 or self.plateau > 10:
			return True 
		return False

	def get_params(self, deep=False):
		return {"lr": self.lr, "batch_size": self.batch_size, "epochs": self.epochs, 
				"input_dim": self.input_dim, "output_dim": self.output_dim, "hidden_dim": self.hidden_dim, 
				"scoring": self.scoring, "threshold": self.threshold, 
				"regularization_param": self.regularization_param}


	def set_params(self, **params):
		self.lr = params.get('lr', self.lr)
		self.batch_size = params.get('batch_size', self.batch_size)
		self.epochs = params.get('epochs', self.epochs)
		self.input_dim = params.get('input_dim', self.input_dim)
		self.output_dim = params.get('output_dim', self.output_dim)
		self.hidden_dim = params.get('hidden_dim', self.hidden_dim)
		self.scoring = params.get('scoring', self.scoring)
		self.threshold = params.get('threshold', self.threshold)
		self.regularization_param = params.get('regularization_param', self.regularization_param)
		return self

class SimpleANNRegressor(SimpleRegressorBase):
	def fit(self, X, y, **kwargs):
		print()
		print('----------------------------------------')
		print('Training with parameters: ' + str(self.get_params()))
		self.model = SimpleANN(self.input_dim, self.output_dim, self.hidden_dim)
		self.min_loss = float("inf")
		self.tol = 0
		self.plateau = 0

		all_training_data = torch.from_numpy(X.astype('float64'))
		all_training_labels = torch.from_numpy(y.astype('float64'))
		data = TensorDataset(all_training_data, all_training_labels)
		loader = DataLoader(data, shuffle=True, batch_size=self.batch_size)

		is_cuda = torch.cuda.is_available()
		if is_cuda:
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")
		self.model.double()
		self.model.to(device)

		#set up loss function and optimizer
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.regularization_param)
		
		# train
		prev_loss = np.inf
		for i in range(self.epochs):
			#train on one pass of data
			self.model.train()
			for inpts, lbls in loader:
				# input should be 3d tensor of shape (batch_size, seq_len, input_dim)
				optimizer.zero_grad()
				self.model.zero_grad()
				inpts, lbls = inpts.to(device), lbls.to(device)
				output = self.model(inpts)

				loss = criterion(output, lbls)
				loss.backward()
				optimizer.step()

			#evaluate total training loss and check stopping condition
			total_loss = 0
			with torch.no_grad():
				self.model.eval()	
				output = self.model(all_training_data)
				total_loss = criterion(output, all_training_labels).item()
				print("Epoch: {}/{}...".format(i+1, self.epochs), "Loss: {:.6f}...".format(total_loss))
			if self.stop_condition(i, total_loss, prev_loss):
				break
			prev_loss = total_loss
		self.trained_for = i
		return self

	def predict(self, X, **kwargs):
		with torch.no_grad():			
			is_cuda = torch.cuda.is_available()
			if is_cuda:
				device = torch.device("cuda")
			else:
				device = torch.device("cpu")
			self.model.double()
			self.model.to(device)
			self.model.eval()
			
			out = self.model(torch.from_numpy(X.astype('float64')))
			predictions = out.squeeze().tolist()
		return predictions

	def score(self, X, y, **kwargs):
		n = X.shape[0]
		predictions = np.nan_to_num(np.array(self.predict(X, **kwargs), dtype='float64'))
		truth = np.nan_to_num(y)
		if self.scoring == 'r2':
			return r2_score(truth, predictions)
		elif self.scoring == 'mda':
			return mda(truth, predictions)
		elif self.scoring == 'l1':
			return -mean_absolute_error(truth, predictions)
		else:
			return -mean_squared_error(truth, predictions)

class SimpleLSTMRegressor(SimpleRegressorBase):
	# seq len corresponds to how big we want the 'memory window' of our model to be
	def __init__(self, input_dim, output_dim, hidden_dim=512, n_layers=1, lr=0.05, batch_size=1, seq_len = 5, epochs = 50, threshold=1e-5, 
					clip = 5, scoring='mse', regularization_param=0):
		super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, lr=lr, 
					batch_size=batch_size, epochs=epochs, threshold=threshold, regularization_param=0)
		self.sequence_length = seq_len
		self.clip = clip
		self.n_layers = n_layers

	def get_params(self, deep=False):
		params = super().get_params(deep=deep)
		params['clip'] = self.clip
		params['seq_len'] = self.sequence_length
		params['n_layers'] = self.n_layers
		return params

	def set_params(self, **params):
		self = super().set_params(**params)
		self.clip = params.get('clip', self.clip)
		self.sequence_length = params.get('seq_len', self.sequence_length)
		self.n_layers = params.get('n_layers', self.n_layers)
		return self

	def fit(self, X, y, **kwargs):
		print()
		print('----------------------------------------')
		print('Training with parameters: ' + str(self.get_params()))
		self.model = SimpleLSTM(self.input_dim, self.output_dim, self.hidden_dim, self.n_layers)
		self.min_loss = float("inf")
		self.tol = 0
		self.plateau = 0

		# the last column of X is the time_index column, which we dont want to train on
		# we only include it because our sampler depends on it
		all_training_data = torch.from_numpy(X[:, :-1].astype('float64'))
		all_training_labels = torch.from_numpy(y.astype('float64'))

		sampler = TimeseriesSampler(X[:, -1:].squeeze().astype('int'), window_size=self.sequence_length, shuffle=True)

		windows = sampler.windows
		n = len(windows)

		loader = [] # will contain tuples of (inpt, lbls)

		for i in range(0, n, self.batch_size):
			# get relevant sequences
			window_batch = []
			for j in range(i, min(i+self.batch_size, n)):
				window_batch.append(windows[j])

			# now local_windows has dimensions batch_size x sequence_length
			# we want to process so that all input tensors are sequence_length x batch_size x hidden_dim
			input_batch = []
			label_batch = []
			local_batch_size = len(window_batch)
			for sequence_index in range(self.sequence_length):
				for w in window_batch:
					input_batch.append(all_training_data[w[sequence_index]])
					label_batch.append(all_training_labels[w[sequence_index]])
			inpt = torch.cat(input_batch).view(self.sequence_length, local_batch_size, -1).double()
			lbl = torch.cat(label_batch).view(self.sequence_length, local_batch_size, -1).double()
			loader.append((inpt, lbl))

		
		is_cuda = torch.cuda.is_available()
		if is_cuda:
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")
		self.model.double()
		self.model.to(device)

		#set up loss function and optimizer
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.regularization_param)
		
		# train
		prev_loss = np.inf
		for i in range(self.epochs):
			#train on one pass of data
			self.model.train()
			for inpts, lbls in loader:
				# input should be 3d tensor of shape (batch_size, seq_len, input_dim)
				optimizer.zero_grad()
				h = self.model.init_hidden(inpts.size(1))
				h = tuple([e.to(device).data for e in h])
				inpts, lbls = inpts.to(device), lbls.to(device)
				self.model.zero_grad()
				output, h = self.model(inpts, h)

				# lbls shape is batch_size x output_dim
				# output shape is batch_size x output_dim?
				loss = criterion(output, lbls[-1])
				loss.backward()
				nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
				optimizer.step()
			# print("Epoch: {}/{}...".format(i+1, self.epochs))

			#evaluate total training loss and check stopping condition
			total_loss = 0
			with torch.no_grad():
				self.model.eval()	
				for inpts, lbls in loader:
					# input should be 3d tensor of shape (batch_size, seq_len, input_dim)
					h = self.model.init_hidden(inpts.size(1))
					h = tuple([e.to(device).data for e in h])
					inpts, lbls = inpts.to(device), lbls.to(device)
					output, h = self.model(inpts, h)
					total_loss += criterion(output, lbls[-1]).item() * inpts.size(0)
				total_loss = total_loss/len(loader)
				print("Epoch: {}/{}...".format(i+1, self.epochs), "Loss: {:.6f}...".format(total_loss))
			if self.stop_condition(i, total_loss, prev_loss):
				break
			prev_loss = total_loss
		self.trained_for = i
		return self

	def predict(self, X, **kwargs):
		with torch.no_grad():
			data = torch.from_numpy(X[:, :-1].astype('float64'))

			sampler = TimeseriesSampler(X[:, -1:].squeeze().astype('int'), window_size=self.sequence_length, shuffle=False)

			is_cuda = torch.cuda.is_available()
			if is_cuda:
				device = torch.device("cuda")
			else:
				device = torch.device("cpu")
			self.model.double()
			self.model.to(device)
			self.model.eval()
			
			predictions = []
			for window in sampler:
				inp = data[window]
				inp = inp.view(self.sequence_length, 1, self.input_dim)
				h = self.model.init_hidden(1)
				h = tuple([e.to(device).data for e in h])
				inp = inp.to(device)
				out, h = self.model(inp, h)
				predictions.append(out.squeeze().item())
		return predictions

	def score(self, X, y, **kwargs):
		n = X.shape[0]
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






