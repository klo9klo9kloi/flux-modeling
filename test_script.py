from autoencoder import SimpleAutoencoder
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import torch.nn as nn

lr = 0.05
batch_size = 10
epochs = 10
hidden_dim = 6 # should maybe automate to be equal to the input_dim of lstm_model
validation_size = 0.25
state_dict_path = './autoencoder_state_dict.pt'

def train_autoencoder(X):
	#expected shape (batch_size, cell_state_hidden_dim)
	n = X.shape[0]
	data = TensorDataset(X, X)
	model = SimpleAutoencoder(X.shape[1], hidden_dim)

	train_data, val_data = random_split(data, [int(n*(1-validation_size)), int(n*0.25)])

	t_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
	v_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)

	is_cuda = torch.cuda.is_available()
	if is_cuda:
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")
	model.to(device)

	#set up loss function and optimizer
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	min_v_loss = float('inf')

	for i in range(epochs):
		model.train()
		for inpts, lbls in t_loader:
			optimizer.zero_grad()
			model.zero_grad()
			inpts, lbls = inpts.to(device), lbls.to(device)
			output, _ = model(inpts)
			loss = criterion(output, lbls)
			loss.backward()
			optimizer.step()
		model.eval()
		v_loss = 0
		for inpts, lbls in v_loader:
			inpts, lbls = inpts.to(device), lbls.to(device)
			output, _ = model(inpts)
			v_loss += criterion(output, lbls).item()
		v_loss = np.mean(v_loss)
		print("Epoch: {}/{}...".format(i+1, epochs), "Avg Val Loss: {:.6f}...".format(v_loss))
		if v_loss < min_v_loss:
			min_v_loss = v_loss
			torch.save(model.state_dict(), state_dict_path)
			print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_v_loss, v_loss))
	return torch.load_state_dict(torch.load(state_dict_path))