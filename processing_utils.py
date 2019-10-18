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

working_directory = '/home/jszou/cs/keenan/'
fluxnet_directory = 'fluxnet/'
visualizations_directory = 'viz'
training_output_directory = 'out'

granularity_to_string = {'YY' : 'Years', 'MM': 'Months', 'WW': 'Weeks', 'DD': 'Days', 'HH': 'Hours'}

def get_zip_name(site_name, year_range, something):
    return working_directory + fluxnet_directory + 'FLX_' + site_name + '_FLUXNET2015_FULLSET_' \
            + year_range + '_' + something + '.zip'

# granularity = 'YY', 'MM', 'WW', 'DD', 'HH' for yearly, monthly, weekly, daily, hourly respectively
def preprocess(site_name, set_type, year_range, something, granularity):
    zf = zipfile.ZipFile(get_zip_name(site_name, year_range, something))
    filename = "FLX_" + site_name + "_FLUXNET2015_" + set_type + "_" + granularity + "_" \
                + year_range + "_" + something + ".csv"
    print('Loading: ' + filename)
    frame = pd.read_csv(zf.open(filename), dtype=str)
    frame['time_index'] = frame.index
    
    print("Total rows: " + str(len(frame.index)))
    print()
    zf.close()
    return frame

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
    offset = sequence_length + 1
    train_len = len(train_pred)
    test_len = len(test_pred)
    x = range(len(ground_truth))

    # have to fill some nans. the first seq_len entries for train will be empty, and the first seq len entries for test will be empty
    # since they are used for memory
    train_line = ([np.nan] * sequence_length) + train_pred + ([np.nan] * (test_len+sequence_length))
    test_line = ([np.nan] * (train_len + 2*sequence_length)) + test_pred

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
    if not os.path.exists(visualizations_directory):
        os.makedirs(visualizations_directory)
    plt.savefig(visualizations_directory + '/' + site_name + '_predictions.png')

    plt.figure()
    total_pred = ([np.nan] * sequence_length) + train_pred + ([np.nan] * sequence_length) + test_pred
    plt.title('Residual Graph for ' + site_name)
    plt.ylabel('Residual')
    plt.xlabel(granularity_to_string[granularity] + ' since ' + start_date)
    plt.scatter(x=x, y=(ground_truth - total_pred))
    plt.savefig(visualizations_directory + '/' + site_name + '_residuals.png')

def generate_file_output(output_strings, site_name):
    if not os.path.exists(training_output_directory):
        os.makedirs(training_output_directory)

    with open(training_output_directory + '/' + site_name + '_out.txt', 'w') as f:
        for output in output_strings:
            f.write(output + '\n')
        f.close()


