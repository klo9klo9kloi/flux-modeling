import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import zipfile
import time
import glob
import re
import ast
import torch
from datetime import datetime, timedelta
from torch.utils.data import TensorDataset, DataLoader
from models import TimeseriesSampler
from scipy.io import loadmat
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

working_directory = os.getcwd()
fAPAR_directory = 'modisfAPAR'
fAPAR_VAR_NAME = 'avg_fAPAR_interpol'
mat_naming_convention = '_MOD15A2H_Fpar_500m.mat'

granularity_to_string = {'YY' : 'Years', 'MM': 'Months', 'WW': 'Weeks', 'DD': 'Days', 'HH': 'Hours'}

def get_training_params(path_to_file):
    """Reads in parameters for model training from the specified configuration file
    and returns a key-value mapping.

    Keyword arguments:
    path_to_file -- relative path to configuration file (str)
    """
    train_params = {}
    with open(path_to_file) as f:
        model_type = f.readline().rstrip()
        hyperparameter_grid = ast.literal_eval(f.readline())
        num_folds = int(f.readline())
        target_variables = ast.literal_eval(f.readline())
        backup_variables = ast.literal_eval(f.readline())
        labels = ast.literal_eval(f.readline())
        granularity = f.readline().rstrip()
        val_size = float(f.readline())
        test_size = float(f.readline())
        offset = int(f.readline())
        out_dir = f.readline().rstrip()
        viz_dir = f.readline().rstrip()
        
        train_params['model_type'] = model_type
        train_params['hyperparameter_grid'] = hyperparameter_grid
        train_params['k'] = num_folds
        train_params['target_variables'] = target_variables
        train_params['backup_variables'] = backup_variables
        train_params['labels'] = labels
        train_params['granularity'] = granularity
        train_params['val_size'] = val_size
        train_params['test_size'] = test_size
        train_params['offset'] = offset
        train_params['out'] = out_dir
        train_params['viz'] = viz_dir
    return train_params

def get_zip_name(target_dir, site_name, year_range, something):
    """Returns the absolute path to target zip file."""
    return working_directory + '/' + target_dir + '/FLX_' + site_name + '_FLUXNET2015_FULLSET_' \
            + year_range + '_' + something + '.zip'

def load_csv_from_zip(target_dir, site_name, set_type, year_range, something, granularity):
    """Loads in data from target csv file."""
    zf = zipfile.ZipFile(get_zip_name(target_dir, site_name, year_range, something))
    filename = "FLX_" + site_name + "_FLUXNET2015_" + set_type + "_" + granularity + "_" \
                + year_range + "_" + something + ".csv"
    print('Loading: ' + filename)
    frame = pd.read_csv(zf.open(filename), dtype=str)
    zf.close()
    return frame

def preprocess(target_dir, site_name, set_type, year_range, something, granularity, target_variables, backup_variables, labels, debug_output, offset=0):
    """Top-level function for preprocessing data from a specified csv file.

    Keyword arguments:
    target_dir -- relative path to directory containing desired data (str)
    site_name -- FLUXNET site name (str)
    set_type -- FULLSET or SUBSET (str)
    year_range -- range of years formated as YYYY-YYYY (str)
    something -- trailing numbers for identifying zip files (str)
    granularity -- YY, MM, WW, DD, HH for yearly, monthly, weekly, daily, hourly respectively (str)
    target_variables -- list of FLUXNET variables to use as predictors (List[str])
    backup_variables -- mapping of back-up of FLUXNET variables to use (dict)
    labels -- list of target FLUXNET response variables (List[str])
    debug_output -- print helpful output for debugging (bool)
    offset -- days to offset training labels by; used for forecasting (int)
    """
    frame = load_csv_from_zip(target_dir, site_name, set_type, year_range, something, granularity)
    print("Total rows: " + str(len(frame.index)))
    print()
    
    frame['time_index'] = frame.index

    if offset:
       # set up data so that label to be predicted is the value x days later
        for label in labels:
            frame[label+'_train'] = frame[label].iloc[offset:].reset_index()[label]
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

    frame = pd.merge(frame, fpar_frame, how='inner', on='TIMESTAMP')
    variables.append(fAPAR_VAR_NAME)

    # we do this at the end to ensure its the last one in the list for processing purposes
    variables.append('time_index')
    print("Total rows for training: " + str(len(frame.index)))
    return frame, variables

def validate_variable(data, variable):
    """Returns whether the desired variable should be used for training."""
    return (variable in data.columns) and (data[variable].value_counts().get('-9999', 0) <= len(data.index)/2)

def validate_frame(df, variables):
    """Returns whether all listed variables are valid within the frame."""
    print(len(df.index))
    assert(len(df.index) >= 100)
    for v in variables:
        if df[v].value_counts().get('-9999', 0) != 0:
            return False
    return True

def get_zip_info(target_dir):
    """Returns zip file information from target directory.

    Keyword arguments:
    target_dir -- relative path to directory (str)
    """
    all_zip_files = glob.glob(working_directory + '/' + target_dir + "/*.zip")
    regex_tuples = []
    for zf in all_zip_files:
        m = re.search('(\w+)/FLX_([^_]+)_FLUXNET2015_(\w+)_(\d+\-\d+)_(\d-\d)', zf)
        regex_tuples.append(m.groups())
    return regex_tuples

def get_mat_info():
    """Returns all relevant mat filenames from fAPAR_directory."""
    all_mat_files = glob.glob(working_directory + '/' + fAPAR_directory + "/*.mat")
    file_names = []
    for mat in all_mat_files:
        file_names.append(mat.split('/')[-1])
    return file_names

def get_avg_fpar_frame(site_name):
    """Returns an interpolated time-series of average fpar for the specified FLUXNET site."""
    mat = loadmat(working_directory + '/modisfAPAR/' + site_name + mat_naming_convention)['FparData']
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

def generate_visualizations(time_index, ground_truth, test_pred, train_pred, granularity, start_date, pred_label, site_name, visualizations_directory):
    """Generates a ground truth vs predictions graph and residuals graph."""
    fig = plt.figure()
    plt.title('Predictions vs. Ground Truth for '+ site_name)
    plt.xlabel(granularity_to_string[granularity] + ' since ' + start_date)
    plt.ylabel(pred_label)
    sns.lineplot(x=time_index, y=ground_truth, label='ground truth')
    sns.lineplot(x=time_index[0: len(train_pred)], y=train_pred, 
                 label='train predictions')
    sns.lineplot(x=time_index[(len(time_index)-len(test_pred)):], y= test_pred,
                 label='test predictions', color='red')

    if not os.path.exists(working_directory + '/' + visualizations_directory + '/' + site_name):
        os.makedirs(working_directory + '/' + visualizations_directory + '/' + site_name)
    plt.savefig(working_directory + '/' + visualizations_directory + '/' + site_name + '/predictions.png')
    plt.close()

    fig = plt.figure()
    total_pred = train_pred + test_pred
    plt.title('Residual Graph for ' + site_name)
    plt.ylabel('Residual')
    plt.xlabel(granularity_to_string[granularity] + ' since ' + start_date)
    plt.scatter(x=time_index, y=(ground_truth - total_pred))
    plt.savefig(working_directory + '/' + visualizations_directory + '/' + site_name + '/residuals.png')
    plt.close()

def generate_file_output(output_strings, site_name, training_output_directory):
    """Write training output to text file."""
    if not os.path.exists(working_directory + '/' + training_output_directory):
        os.makedirs(working_directory + '/' + training_output_directory)

    with open(working_directory + '/' + training_output_directory + '/' + site_name + '_out.txt', 'w') as f:
        for output in output_strings:
            f.write(output + '\n')
        f.close()

def generate_weights_visualization(model, variables, site_name, visualizations_directory):
    """Generate a heatmap for all learned weights of the LSTM."""
    param_list = list(model.model.parameters())
    dim = model.model.hidden_dim
    input_weights = param_list[0].data.cpu()
    ii_weights = input_weights[:dim]
    if_weights = input_weights[dim:dim*2]
    ig_weights = input_weights[dim*2:dim*3]
    io_weights = input_weights[dim*3:dim*4]

    if not os.path.exists(working_directory + '/' + visualizations_directory + '/' + site_name):
        os.makedirs(working_directory + '/' + visualizations_directory)
    plt.rc("axes", titlesize=18)

    fig = plt.figure()
    sns.heatmap(ii_weights, xticklabels=variables)
    plt.title('Input Gate Weights for Input')
    fig.set_size_inches(16,14)
    plt.savefig(working_directory + '/' + visualizations_directory + '/' + site_name + '/ii_weights.png')
    plt.close()

    fig = plt.figure()
    sns.heatmap(if_weights, xticklabels=variables)
    plt.title('Forget Gate Weights for Input')
    fig.set_size_inches(16,14)
    plt.savefig(working_directory + '/' + visualizations_directory + '/' + site_name + '/if_weights.png')
    plt.close()

    fig = plt.figure()
    sns.heatmap(ig_weights, xticklabels=variables)
    plt.title('Cell State Weights for Input')
    fig.set_size_inches(16,14)
    plt.savefig(working_directory + '/' + visualizations_directory + '/' + site_name + '/ig_weights.png')
    plt.close()

    fig = plt.figure()
    sns.heatmap(io_weights, xticklabels=variables)
    plt.title('Output Gate Weights for Input')
    fig.set_size_inches(16,14)
    plt.savefig(working_directory + '/' + visualizations_directory + '/' + site_name + '/io_weights.png')
    plt.close()


def generate_variability_graph(zip_info, training_output_directory, visualizations_directory, model_type, extras=""):
    """Generate a ground truth vs prediction graph with standard error bars."""
    site_name = zip_info[1]

    path = training_output_directory + '/predictions/' + site_name + '.txt'

    df = pd.read_csv(path)
    original_data = load_csv_from_zip(*zip_info, 'DD')

    time_index = df.iloc[:,:-1].columns.values.astype('float').astype('int')
    ground_truth = original_data.iloc[time_index]['GPP_NT_VUT_REF'].astype('float')
    predictions = df.iloc[:, :-1].melt()
    predictions['variable'] = predictions['variable'].astype('float').astype('int')

    fig = plt.figure()
    sns.lineplot(x=ground_truth.index, y=ground_truth.values, label='ground truth')
    palette = sns.color_palette()
    ax = sns.lineplot(x="variable", y="value", data=predictions,
                 label='predictions', ci="sd", err_style="band", color=palette[4])
    ax.grid(False)
    plt.title(model_type.upper() + ' Predictions for '+ site_name + " " + extras)
    plt.xlabel("Days since " + original_data['TIMESTAMP'].iloc[0])
    plt.ylabel('GPP_NT_VUT_REF')
    # plt.show()
    if not os.path.exists(visualizations_directory + '/' + site_name):
        os.makedirs(visualizations_directory + '/' + site_name)
    plt.savefig(visualizations_directory + '/' + site_name + '/prediction_variability.png')
    plt.close()

def generate_r2_chart(zip_infos, training_output_directory, visualizations_directory, model_type, extras=""):
    """Graph average model R^2 score with standard error bars."""
    site_names = []
    scores = []
    for zf in zip_infos:
        path = training_output_directory + '/predictions/' + zf[1] + '.txt'

        df = pd.read_csv(path)
        scores += list(df.iloc[:, -1])
        site_names += ([zf[1]] * len(df.index))

    score_frame = pd.DataFrame({"site": site_names, "score": scores})
    fig = plt.figure()
    sns.set(style="whitegrid")
    palette = sns.color_palette()
    ax = sns.barplot(x="site", y="score", data=score_frame, ci="sd", palette={"AU-Gin": palette[3], 'CA-NS3': palette[1], 'CZ-BK1': palette[2], 'US-Ton': palette[4]})
    ax.grid(False)
    plt.title(model_type.upper() + ' performance across sites ' + extras)
    plt.xlabel("Site")
    plt.ylabel("Coefficient of Determination (R^2)")
    plt.ylim(0, 0.9)
    # plt.show()
    if not os.path.exists(visualizations_directory):
        os.makedirs(visualizations_directory)
    plt.savefig(visualizations_directory + '/r2_across_sites.png')
    plt.close()

def generate_generalizability_chart(fluxnet_site_type, training_output_directory, visualizations_directory, model_type, extras=""):
    """Graph the results of generalizability testing with standard error bars."""
    path = working_directory + '/' + training_output_directory + '/' + fluxnet_site_type

    all_output_files = glob.glob(path + "/*_generalizability_test.txt")

    all_output_frame = pd.DataFrame()

    for output_file in all_output_files:
        df = pd.read_csv(output_file)
        all_output_frame = pd.concat((all_output_frame, df))

    order = ['AU-Ade', 'AU-How', 'AU-Gin', 'AU-RDF', 'US-Ton', 'US-SRM']

    fig = plt.figure()
    sns.set(style="whitegrid")
    palette = sns.color_palette()
    palette[2] = palette[3]
    palette[0:2] = palette[5:7]
    palette[3] = palette[8]
    palette[5:6] = palette[9:10]
    ax = sns.barplot(x="trained_on", y="r2", ci="sd", hue="site", palette=palette, hue_order=order, order=order, data=all_output_frame)
    ax.grid(False)
    plt.legend(title="Evaluated on", loc=1)
    plt.xlabel("Trained on")
    plt.ylabel("Coefficient of Determination (R^2)")
    plt.ylim(0, 0.7)
    plt.title(model_type.upper() + " generalizability to other " + fluxnet_site_type + "-type sites " + extras)
    fig.set_size_inches(10, 8)
    # plt.show()

    viz_path = working_directory + '/' + visualizations_directory + '/' + fluxnet_site_type

    if not os.path.exists(viz_path):
        os.makedirs(viz_path)
    plt.savefig(viz_path + '/generalizability.png')
    plt.close()

def generate_universality_chart(fluxnet_site_type, training_output_directory, visualizations_directory, model_type, extras=""):
    """Graph the results of universality testing with standard error bars."""
    path = working_directory + '/' + training_output_directory + '/' + fluxnet_site_type
    df = pd.read_csv(path + '/universiality_test.txt')

    order = ['AU-Ade', 'AU-How', 'AU-Gin', 'AU-RDF', 'US-Ton', 'US-SRM']

    fig = plt.figure()
    sns.set(style="whitegrid")
    palette = sns.color_palette()
    palette[2] = palette[3]
    palette[0:2] = palette[5:7]
    palette[3] = palette[8]
    palette[5:6] = palette[9:10]
    ax = sns.barplot(x="site", y="r2", ci="sd", palette=palette, order=order, data=df)
    ax.grid(False)
    plt.xlabel("Evaluation Site")
    plt.ylabel("Coefficient of Determination (R^2)")
    plt.ylim(0, 0.8)
    plt.title("All-site " + model_type.upper() + " model performance per site " + extras)
    fig.set_size_inches(10, 8)
    # plt.show()

    viz_path = working_directory + '/' + visualizations_directory + '/' + fluxnet_site_type

    if not os.path.exists(viz_path):
        os.makedirs(viz_path)
    plt.savefig(viz_path + '/universality.png')
    plt.close()


def generate_weight_variance_chart(zip_infos, training_output_directory, visualizations_directory, extras=""):
    """Graph the results of weight variance quantification with standard error bars."""
    path = working_directory + '/' + training_output_directory + '/weights'

    all_output_frame = pd.DataFrame()

    for zf in zip_infos:
        df = pd.read_csv(path + '/' + zf[1] + '_weight_variance.txt')

        #normalize to 0-1 scale using max weight across variables
        num_vars = len(df['target_variable'].unique())
        new_vals = []
        for i in range(0, len(df.index), num_vars):
            absolute_sums = df.loc[i:i+num_vars-1, 'variability']
            new_vals += list(absolute_sums/max(absolute_sums))

        df['variability'] = new_vals
        all_output_frame = pd.concat((all_output_frame, df))

    order = ['AU-Gin', 'CA-NS3', 'CZ-BK1', 'US-Ton']
    
    fig = plt.figure()

    sns.set(style="whitegrid")
    palette = sns.color_palette()
    ax = sns.barplot(x="target_variable", y="variability",ci="sd", hue="site", 
                        palette={"AU-Gin": palette[3], 'CA-NS3': palette[1], 'CZ-BK1': palette[2], 'US-Ton': palette[4]}, 
                        hue_order=order, data=all_output_frame[all_output_frame['weight_type'] == 'input-input'])
    plt.xticks([0,1,2,3,4,5,6,7], ['TA', 'SW_IN', 'P', 'WS', 'VPD', 'CO2', 'SWC', 'fAPAR'])
    ax.grid(False)
    plt.legend(title="Site", loc=1)
    plt.xlabel("Variable")
    plt.ylabel("Weight Variance (Sum of Absolute Weights)")
    # # plt.ylim(0, 0.7)
    plt.title("LSTM Input-Input Gate Weight Variability across sites " + extras)
    fig.set_size_inches(10, 8)
    # plt.show()

    viz_path = working_directory + '/' + visualizations_directory

    if not os.path.exists(viz_path):
        os.makedirs(viz_path)
    plt.savefig(viz_path + '/ii_weight_variance.png')
    plt.close()


    fig = plt.figure()
    ax = sns.barplot(x="target_variable", y="variability",ci="sd", hue="site", 
                        palette={"AU-Gin": palette[3], 'CA-NS3': palette[1], 'CZ-BK1': palette[2], 'US-Ton': palette[4]}, 
                        hue_order=order, data=all_output_frame[all_output_frame['weight_type'] == 'input-cell state'])
    plt.xticks([0,1,2,3,4,5,6,7], ['TA', 'SW_IN', 'P', 'WS', 'VPD', 'CO2', 'SWC', 'fAPAR'])
    ax.grid(False)
    plt.legend(title="Site", loc=1)
    plt.xlabel("Variable")
    plt.ylabel("Weight Variance (Sum of Absolute Weights)")
    # # plt.ylim(0, 0.7)
    plt.title("LSTM Input-Cell State Gate Weight Variability across sites " + extras)
    fig.set_size_inches(10, 8)
    plt.savefig(viz_path + '/ig_weight_variance.png')
    plt.close()

    fig = plt.figure()
    ax = sns.barplot(x="target_variable", y="variability",ci="sd", hue="site", 
                        palette={"AU-Gin": palette[3], 'CA-NS3': palette[1], 'CZ-BK1': palette[2], 'US-Ton': palette[4]}, 
                        hue_order=order, data=all_output_frame[all_output_frame['weight_type'] == 'input-forget'])
    plt.xticks([0,1,2,3,4,5,6,7], ['TA', 'SW_IN', 'P', 'WS', 'VPD', 'CO2', 'SWC', 'fAPAR'])
    ax.grid(False)
    plt.legend(title="Site", loc=1)
    plt.xlabel("Variable")
    plt.ylabel("Weight Variance (Sum of Absolute Weights)")
    # # plt.ylim(0, 0.7)
    plt.title("LSTM Input-Forget Gate Weight Variability across sites " + extras)
    fig.set_size_inches(10, 8)
    plt.savefig(viz_path + '/if_weight_variance.png')
    plt.close()

    fig = plt.figure()
    ax = sns.barplot(x="target_variable", y="variability", ci="sd", hue="site", 
                        palette={"AU-Gin": palette[3], 'CA-NS3': palette[1], 'CZ-BK1': palette[2], 'US-Ton': palette[4]}, 
                        hue_order=order, data=all_output_frame[all_output_frame['weight_type'] == 'input-output'])
    plt.xticks([0,1,2,3,4,5,6,7], ['TA', 'SW_IN', 'P', 'WS', 'VPD', 'CO2', 'SWC', 'fAPAR'])
    ax.grid(False)
    plt.legend(title="Site", loc=1)
    plt.xlabel("Variable")
    plt.ylabel("Weight Variance (Sum of Absolute Weights)")
    # # plt.ylim(0, 0.7)
    plt.title("LSTM Input-Output Gate Weight Variability across sites " + extras)
    fig.set_size_inches(10, 8)
    plt.savefig(viz_path + '/io_weight_variance.png')
    plt.close()


