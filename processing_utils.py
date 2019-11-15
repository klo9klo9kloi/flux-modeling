import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import zipfile
from datetime import datetime, timedelta
import time
from sklearn.model_selection import KFold
import glob
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
from lstm_model import TimeseriesSampler
from test_script import train_autoencoder
from scipy.io import loadmat
from scipy.interpolate import interp1d

working_directory = '/home/jszou/cs/keenan/'
fluxnet_directory = 'testnet/'
fAPAR_directory = 'modisfAPAR/'
fAPAR_VAR_NAME = 'avg_fAPAR_interpol'
mat_naming_convention = '_MOD15A2H_Fpar_500m.mat'
visualizations_directory = 'viz'
training_output_directory = 'out'

granularity_to_string = {'YY' : 'Years', 'MM': 'Months', 'WW': 'Weeks', 'DD': 'Days', 'HH': 'Hours'}

def get_zip_name(site_name, year_range, something):
    return working_directory + fluxnet_directory + 'FLX_' + site_name + '_FLUXNET2015_FULLSET_' \
            + year_range + '_' + something + '.zip'

# granularity = 'YY', 'MM', 'WW', 'DD', 'HH' for yearly, monthly, weekly, daily, hourly respectively
def preprocess(site_name, set_type, year_range, something, granularity, target_variables, backup_variables, labels, debug_output):
    zf = zipfile.ZipFile(get_zip_name(site_name, year_range, something))
    filename = "FLX_" + site_name + "_FLUXNET2015_" + set_type + "_" + granularity + "_" \
                + year_range + "_" + something + ".csv"
    print('Loading: ' + filename)
    frame = pd.read_csv(zf.open(filename), dtype=str)
    
    print("Total rows: " + str(len(frame.index)))
    print()
    zf.close()

    frame['time_index'] = frame.index
    # set up data so that label to be predicted is the value 1 day later
    for label in labels:
        frame[label+'_train'] = frame[label].iloc[1:].reset_index()[label]

    # print(frame.head())
    # print(frame.iloc[-1])
    # print(frame.head())
    # print(frame.iloc[-1])

    variables = []
    for v in target_variables:
        if validate_variable(frame, v):
            variables.append(v)
            frame = frame[frame[v] != '-9999'] #causing future warnings
        elif v in backup_variables and validate_variable(frame, backup_variables[v]):
            variables.append(backup_variables[v])
            frame = frame[frame[backup_variables[v]] != '-9999'] #causing future warnings
            debug_output.append('Using backup variable for ' + v)
        else:
            debug_output.append('Variable ' + v + ' is not used during training for this dataset, either because it is missing or has missing values.')

    assert(validate_frame(frame, variables) == True)

    # add remote sensing data
    fpar_frame = get_avg_fpar_frame(site_name)

    frame = pd.merge(frame, fpar_frame, how='left', on='TIMESTAMP')
    variables.append(fAPAR_VAR_NAME)

    
    variables.append('time_index')

    return frame, variables

# data: DataFrame
# variable: String
def validate_variable(data, variable):
    return (variable in data.columns)

def validate_frame(df, variables):
    assert(len(df.index) >= 100)
    for v in variables:
        if df[v].value_counts().get('-9999', 0) != 0:
            return False
    return True

def get_zip_info():
    all_zip_files = glob.glob(working_directory + fluxnet_directory + "*.zip")
    regex_tuples = []
    for zf in all_zip_files:
        m = re.search('FLX_([^_]+)_FLUXNET2015_(\w+)_(\d+\-\d+)_(\d-\d)', zf)
        regex_tuples.append(m.groups())
    return regex_tuples

def get_mat_info():
    all_mat_files = glob.glob(working_directory + fAPAR_directory + "*.mat")
    file_names = []
    for mat in all_mat_files:
        file_names.append(mat.split('/')[-1])
    return file_names

def get_avg_fpar_frame(site_name):
    x = loadmat('modisfAPAR/' + site_name + mat_naming_convention)['FparData']
    df = pd.DataFrame(data=x, columns=['year', 'DOY', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    x = list(range(0, len(df.index)*8, 8))
    y = np.mean(df.iloc[:, 2:], axis=1)

    timestamp = df['year'].astype('str') + df['DOY'].astype('str')
    start_date = datetime.strptime(timestamp[0], '%Y%j')
    end_date = datetime.strptime(timestamp.iloc[-1], '%Y%j')

    delta = end_date - start_date
    timestamp = []
    for i in range(delta.days+1):
        day = start_date + timedelta(days=i)
        timestamp.append(day.strftime('%Y%m%d'))

    f = interp1d(x, y)

    xnew = list(range(delta.days+1))
    # plt.plot(x, y, 'o', xnew, f(xnew))
    # plt.show()
    ynew = f(xnew)

    new_frame = pd.DataFrame({'TIMESTAMP': timestamp, fAPAR_VAR_NAME: ynew})
    return new_frame

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

def generate_visualizations(model, time_index, ground_truth, test_set, train_set, sequence_length, granularity, start_date, pred_label, site_name):
    y_pred = model.predict(test_set) #can call predict directly because refit=True
    train_pred = model.predict(train_set)

    # have to fill some nans. the first seq_len entries for train will be empty because used for lstm memory. 
    # the last training prediction will bleed into the first entry used for memory by test predictions.
    # the last test prediction is beyond the scope of our ground truth since it predicts the day after, so we choose to not plot it for now

    # train_line = ([np.nan] * (sequence_length) ) + train_pred + ([np.nan] * (test_len+sequence_length-2))
    # test_line = ([np.nan] * (train_len + 2*(sequence_length) - 1 )) + test_pred[:-1]

    # print(time_index)
    # print(len(time_index))
    # print(len(y_pred))
    # print(len(train_pred))

    plt.figure()
    plt.title('Predictions vs. Ground Truth for '+ site_name)
    #TODO: parse datetime from start_date
    plt.xlabel(granularity_to_string[granularity] + ' since ' + start_date)
    plt.ylabel(pred_label)
    sns.lineplot(x=time_index, y=ground_truth, label='ground truth')
    sns.lineplot(x=time_index[sequence_length: (sequence_length + len(train_pred))], y=train_pred, 
                 label='train predictions')
    sns.lineplot(x=time_index[2*sequence_length+len(train_pred)-1:], y= y_pred[:-1],
                 label='test predictions', color='red')

    #TODO: generate visualization of model weights to see what its learning
    if not os.path.exists(visualizations_directory + '/' + site_name):
        os.makedirs(visualizations_directory + '/' + site_name)
    plt.savefig(visualizations_directory + '/' + site_name + '/predictions.png')

    plt.figure()
    total_pred = ([np.nan] * (sequence_length) ) + train_pred + ([np.nan] * (sequence_length-1) ) + y_pred[:-1]
    plt.title('Residual Graph for ' + site_name)
    plt.ylabel('Residual')
    plt.xlabel(granularity_to_string[granularity] + ' since ' + start_date)
    plt.scatter(x=time_index, y=(ground_truth - total_pred))
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
    plt.title('Output Gate Weights for Input')
    plt.savefig(visualizations_directory + '/' + site_name + '/io_weights.png')




