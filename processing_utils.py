import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import zipfile
from datetime import datetime
import time
from sklearn.model_selection import KFold
import glob
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
from lstm_model import TimeseriesSampler
from test_script import train_autoencoder

working_directory = '/home/jszou/cs/keenan/'
fluxnet_directory = 'testnet/'
visualizations_directory = 'viz'
training_output_directory = 'out'

granularity_to_string = {'YY' : 'Years', 'MM': 'Months', 'WW': 'Weeks', 'DD': 'Days', 'HH': 'Hours'}

def get_zip_name(site_name, year_range, something):
    return working_directory + fluxnet_directory + 'FLX_' + site_name + '_FLUXNET2015_FULLSET_' \
            + year_range + '_' + something + '.zip'

# granularity = 'YY', 'MM', 'WW', 'DD', 'HH' for yearly, monthly, weekly, daily, hourly respectively
def preprocess(site_name, set_type, year_range, something, granularity, labels):
    zf = zipfile.ZipFile(get_zip_name(site_name, year_range, something))
    filename = "FLX_" + site_name + "_FLUXNET2015_" + set_type + "_" + granularity + "_" \
                + year_range + "_" + something + ".csv"
    print('Loading: ' + filename)
    frame = pd.read_csv(zf.open(filename), dtype=str)
    frame['time_index'] = frame.index
    
    print("Total rows: " + str(len(frame.index)))
    print()
    zf.close()

    # set up data so that label to be predicted is the value 1 day later
    for label in labels:
        frame[label+'_train'] = frame[label].iloc[1:].reset_index()[label]
    return frame

# data: DataFrame
# variable: String
def validate_variable(data, variable):
    return (variable in data.columns) and (data[variable].value_counts().get("-9999", 0) == 0) 

def get_zip_info():
    all_zip_files = glob.glob(working_directory + fluxnet_directory + "*.zip")
    regex_tuples = []
    for zf in all_zip_files:
        m = re.search('FLX_([^_]+)_FLUXNET2015_(\w+)_(\d+\-\d+)_(\d-\d)', zf)
        regex_tuples.append(m.groups())
    return regex_tuples

def split_dataset(df, train_prop, k):
    n = len(df.index)
    split_index = int(n*train_prop)
    train_set = df.iloc[:split_index]

    kf = KFold(n_splits=k)
    train = []
    val = []
    # these indexes are a list of indices
    for train_index, val_index in kf.split(train_set):
        train.append(train_index)
        val.append(val_index)
    return train, val, split_index

def generate_visualizations(ground_truth, train_pred, test_pred, sequence_length, granularity, start_date, pred_label, site_name):
    train_len = len(train_pred)
    test_len = len(test_pred)
    x = range(len(ground_truth))

    # have to fill some nans. the first seq_len entries for train will be empty because used for lstm memory. 
    # the last training prediction will bleed into the first entry used for memory by test predictions.
    # the last test prediction is beyond the scope of our ground truth since it predicts the day after, so we choose to not plot it for now
    train_line = ([np.nan] * (sequence_length) ) + train_pred + ([np.nan] * (test_len+sequence_length-2))
    test_line = ([np.nan] * (train_len + 2*(sequence_length) - 1 )) + test_pred[:-1]

    plt.figure()
    plt.title('Predictions vs. Ground Truth for '+ site_name)
    #TODO: parse datetime from start_date
    plt.xlabel(granularity_to_string[granularity] + ' since ' + start_date)
    plt.ylabel(pred_label)
    sns.lineplot(x=x, y=ground_truth, label='ground truth')
    sns.lineplot(x=x, y=train_line, 
                 label='train predictions')
    sns.lineplot(x=x, y= test_line,
                 label='test predictions', color='red')

    #TODO: generate visualization of model weights to see what its learning
    if not os.path.exists(visualizations_directory + '/' + site_name):
        os.makedirs(visualizations_directory + '/' + site_name)
    plt.savefig(visualizations_directory + '/' + site_name + '/predictions.png')

    plt.figure()
    total_pred = ([np.nan] * (sequence_length) ) + train_pred + ([np.nan] * (sequence_length-1) ) + test_pred[:-1]
    plt.title('Residual Graph for ' + site_name)
    plt.ylabel('Residual')
    plt.xlabel(granularity_to_string[granularity] + ' since ' + start_date)
    plt.scatter(x=x, y=(ground_truth - total_pred))
    plt.savefig(visualizations_directory + '/' + site_name + '/residuals.png')

def generate_file_output(output_strings, site_name):
    if not os.path.exists(training_output_directory):
        os.makedirs(training_output_directory)

    with open(training_output_directory + '/' + site_name + '_out.txt', 'w') as f:
        for output in output_strings:
            f.write(output + '\n')
        f.close()

def generate_weights_visualization(model, variables, site_name):
    param_list = list(model.model.parameters())
    dim = model.model.hidden_dim
    input_weights = param_list[0].data.cpu()
    ii_weights = input_weights[:dim]
    if_weights = input_weights[dim:dim*2]
    ig_weights = input_weights[dim*2:dim*3]
    io_weights = input_weights[dim*3:dim*4]

    if not os.path.exists(visualizations_directory + '/' + site_name):
        os.makedirs(visualizations_directory)
    plt.figure()
    sns.heatmap(ii_weights, xticklabels=variables)
    plt.title('Input Gate Weights for Input')
    plt.savefig(visualizations_directory + '/' + site_name + '/ii_weights.png')

    plt.figure()
    sns.heatmap(if_weights, xticklabels=variables)
    plt.title('Forget Gate Weights for Input')
    plt.savefig(visualizations_directory + '/' + site_name + '/if_weights.png')

    plt.figure()
    sns.heatmap(ig_weights, xticklabels=variables)
    plt.title('Cell State Weights for Input')
    plt.savefig(visualizations_directory + '/' + site_name + '/ig_weights.png')

    plt.figure()
    sns.heatmap(io_weights, xticklabels=variables)
    plt.title('Input Gate Weights for Input')
    plt.savefig(visualizations_directory + '/' + site_name + '/io_weights.png')

#model: nn.Module
#data: np.array
# def generate_cell_state_visualization(model, data):
#     # print(cell_state_data)
#     # should have dim (batch_size, seq_len, hidden_dim)

#     cell_state_data = model.get_cell_state_data(data)
#     print("Cell state data")
#     print(cell_state_data[0].shape)
#     print(len(cell_state_data))

#     X = torch.cat(cell_state_data)
#     trained_model = train_autoencoder(X)

#     #ignore time_index from here on out
#     data = data[:, :-1]
#     min_col_vals = np.min(data, axis=0)
#     max_col_vals = np.max(data, axis=0)
#     rows, cols = data.shape

#     heat_map = []
#     for i in range(cols):
#         ls = np.linspace(min_col_vals[i], max_col_vals[i], num=rows)
#     trained_model.eval()



