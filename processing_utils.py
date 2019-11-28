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
from models import TimeseriesSampler
from test_script import train_autoencoder
from scipy.io import loadmat
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

working_directory = os.getcwd()
fAPAR_directory = 'modisfAPAR'
fAPAR_VAR_NAME = 'avg_fAPAR_interpol'
mat_naming_convention = '_MOD15A2H_Fpar_500m.mat'
visualizations_directory = 'viz'
training_output_directory = 'out'

granularity_to_string = {'YY' : 'Years', 'MM': 'Months', 'WW': 'Weeks', 'DD': 'Days', 'HH': 'Hours'}

def get_zip_name(target_dir, site_name, year_range, something):
    return working_directory + '/' + target_dir + '/FLX_' + site_name + '_FLUXNET2015_FULLSET_' \
            + year_range + '_' + something + '.zip'

def load_csv_from_zip(target_dir, site_name, set_type, year_range, something, granularity):
    zf = zipfile.ZipFile(get_zip_name(target_dir, site_name, year_range, something))
    filename = "FLX_" + site_name + "_FLUXNET2015_" + set_type + "_" + granularity + "_" \
                + year_range + "_" + something + ".csv"
    print('Loading: ' + filename)
    frame = pd.read_csv(zf.open(filename), dtype=str)
    zf.close()
    return frame

# granularity = 'YY', 'MM', 'WW', 'DD', 'HH' for yearly, monthly, weekly, daily, hourly respectively
def preprocess(target_dir, site_name, set_type, year_range, something, granularity, target_variables, backup_variables, labels, debug_output, offset=False):
    frame = load_csv_from_zip(target_dir, site_name, set_type, year_range, something, granularity)
    print("Total rows: " + str(len(frame.index)))
    print()
    
    frame['time_index'] = frame.index

    if offset:
       # set up data so that label to be predicted is the value 1 day later
        for label in labels:
            frame[label+'_train'] = frame[label].iloc[1:].reset_index()[label]
    else:
        for label in labels:
            frame[label+'_train'] = frame[label]

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

    # we do this at the end to ensure its the last one in the list for processing purposes
    variables.append('time_index')

    return frame, variables

# data: DataFrame
# variable: String
def validate_variable(data, variable):
    return (variable in data.columns) and (data[variable].value_counts().get('-9999', 0) <= len(data.index)/2)

def validate_frame(df, variables):
    print(len(df.index))
    assert(len(df.index) >= 100)
    for v in variables:
        if df[v].value_counts().get('-9999', 0) != 0:
            return False
    return True

def get_zip_info(target_dir):
    all_zip_files = glob.glob(working_directory + '/' + target_dir + "/*.zip")
    regex_tuples = []
    for zf in all_zip_files:
        m = re.search('(\w+)/FLX_([^_]+)_FLUXNET2015_(\w+)_(\d+\-\d+)_(\d-\d)', zf)
        regex_tuples.append(m.groups())
    return regex_tuples

def get_mat_info():
    all_mat_files = glob.glob(working_directory + '/' + fAPAR_directory + "/*.mat")
    file_names = []
    for mat in all_mat_files:
        file_names.append(mat.split('/')[-1])
    return file_names

def get_avg_fpar_frame(site_name):
    mat = loadmat('modisfAPAR/' + site_name + mat_naming_convention)['FparData']
    df = pd.DataFrame(data=mat, columns=['year', 'DOY', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])

    timestamp = df['year'].astype('str') + df['DOY'].astype('str')
    start_date = datetime.strptime(timestamp[0], '%Y%j')
    end_date = datetime.strptime(timestamp.iloc[-1], '%Y%j')

    # set up data for interpolation
    x = []
    for i in range(len(df.index)):
        date = datetime.strptime(timestamp.iloc[i], '%Y%j')
        x.append( (date-start_date).days)
    y = np.mean(df.iloc[:, 2:], axis=1)

    # set up timestamps to be associated with interpolated data
    delta = end_date - start_date
    timestamp = []
    for i in range(delta.days+1):
        day = start_date + timedelta(days=i)
        timestamp.append(day.strftime('%Y%m%d'))

    f = interp1d(x, y)

    xnew = list(range(delta.days+1))
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

def generate_visualizations(time_index, ground_truth, test_pred, train_pred, sequence_length, granularity, start_date, pred_label, site_name):
    plt.figure()
    plt.title('Predictions vs. Ground Truth for '+ site_name)
    #TODO: parse datetime from start_date
    plt.xlabel(granularity_to_string[granularity] + ' since ' + start_date)
    plt.ylabel(pred_label)
    sns.lineplot(x=time_index, y=ground_truth, label='ground truth')
    sns.lineplot(x=time_index[0: len(train_pred)], y=train_pred, 
                 label='train predictions')
    sns.lineplot(x=time_index[len(train_pred):], y= test_pred,
                 label='test predictions', color='red')

    if not os.path.exists(visualizations_directory + '/' + site_name):
        os.makedirs(visualizations_directory + '/' + site_name)
    plt.savefig(visualizations_directory + '/' + site_name + '/predictions.png')

    # total_fAPAR = np.append(train_set[:, -2:-1], test_set[:, -2:-1])
    # plt.figure()
    # sns.lineplot(x=time_index, y=total_fAPAR, label="fAPAR0", color="green")
    # plt.show()

    plt.figure()
    total_pred = train_pred + test_pred
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
    plt.rc("axes", titlesize=18)

    fig = plt.figure()
    sns.heatmap(ii_weights, xticklabels=variables)
    plt.title('Input Gate Weights for Input')
    fig.set_size_inches(16,14)
    plt.savefig(visualizations_directory + '/' + site_name + '/ii_weights.png')

    fig = plt.figure()
    sns.heatmap(if_weights, xticklabels=variables)
    plt.title('Forget Gate Weights for Input')
    fig.set_size_inches(16,14)
    plt.savefig(visualizations_directory + '/' + site_name + '/if_weights.png')

    fig = plt.figure()
    sns.heatmap(ig_weights, xticklabels=variables)
    plt.title('Cell State Weights for Input')
    fig.set_size_inches(16,14)
    plt.savefig(visualizations_directory + '/' + site_name + '/ig_weights.png')

    fig = plt.figure()
    sns.heatmap(io_weights, xticklabels=variables)
    plt.title('Output Gate Weights for Input')
    fig.set_size_inches(16,14)
    plt.savefig(visualizations_directory + '/' + site_name + '/io_weights.png')


def generate_variability_graph(zip_info):
    site_name = zip_info[1]

    path = training_output_directory + '/predictions/' + site_name + '.txt'

    df = pd.read_csv(path)
    original_data = load_csv_from_zip(*zip_info, 'DD')

    time_index = df.iloc[:,:-1].columns.values.astype('float').astype('int')
    ground_truth = original_data.iloc[time_index]['GPP_NT_VUT_REF'].astype('float')
    predictions = df.iloc[:, :-1].melt()
    predictions['variable'] = predictions['variable'].astype('float').astype('int')

    plt.figure()
    sns.set(style="darkgrid")
    sns.lineplot(x=ground_truth.index, y=ground_truth.values, label='ground truth', color='red')
    sns.lineplot(x="variable", y="value", data=predictions,
                 label='predictions', color='blue', ci="sd", err_style="band")
    plt.title('LSTM Predictions for '+ site_name + ' (no forecasting)')
    plt.xlabel("Days since " + original_data['TIMESTAMP'].iloc[0])
    plt.ylabel('GPP_NT_VUT_REF')
    # plt.show()
    if not os.path.exists(visualizations_directory + '/' + site_name):
        os.makedirs(visualizations_directory + '/' + site_name)
    plt.savefig(visualizations_directory + '/' + site_name + '/prediction_variability.png')

def generate_r2_chart(zip_infos):
    site_names = []
    scores = []
    for zf in zip_infos:
        path = training_output_directory + '/predictions/' + zf[1] + '.txt'

        df = pd.read_csv(path)
        scores += list(df.iloc[:, -1])
        site_names += ([zf[1]] * len(df.index))

    score_frame = pd.DataFrame({"site": site_names, "score": scores})
    plt.figure()
    sns.set(style="whitegrid")
    sns.barplot(x="site", y="score", data=score_frame, ci="sd")
    plt.title('LSTM Performance across Sites (no forecasting)')
    plt.xlabel("Site")
    plt.ylabel("Coefficient of Determination (R^2)")
    # plt.show()
    if not os.path.exists(visualizations_directory):
        os.makedirs(visualizations_directory)
    plt.savefig(visualizations_directory + '/r2_across_sites.png')

def fix_prediction_data(zip_infos):
    for zf in zip_infos:
        path = training_output_directory + '/predictions/' + zf[1] + '.txt'
        df = pd.read_csv(path)
        original_data = load_csv_from_zip(*zf, 'DD')
        time_index = df.iloc[:,:-1].columns.values.astype('float').astype('int')
        ground_truth = original_data.iloc[time_index]['GPP_NT_VUT_REF'].astype('float')

        for i in range(len(df.index)):
            predictions = df.iloc[i, :-1]
            df.iloc[i, -1] = r2_score(ground_truth, predictions)
        df.to_csv(path, index=False)

def generate_generalizability_chart(fluxnet_site_type, model):
    path = training_output_directory + '/' + fluxnet_site_type + '/generalizability_test.txt'

    df = pd.read_csv(path)
    fig = plt.figure()
    sns.set(style="whitegrid")
    sns.barplot(x="trained_on", y="r2", hue="site", data=df)
    plt.xlabel("Trained On")
    plt.ylabel("Coefficient of Determination (R^2)")
    plt.ylim(0, 1)
    plt.title(model + " Performance Across WSA Sites")
    fig.set_size_inches(10, 8)

    if not os.path.exists(visualizations_directory + '/' + fluxnet_site_type):
        os.makedirs(visualizations_directory + '/' + fluxnet_site_type)
    plt.savefig(visualizations_directory + '/' + fluxnet_site_type + '/generalizability.png')




