from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from processing_utils import preprocess, get_training_params
from estimators import get_model_type
import numpy as np
import pandas as pd
import os
import ast
import torch
def quantify_weight_variability(site_zip_info, path_to_config_file, num_iter):
    config = get_training_params(path_to_config_file)
    train_labels = [l + '_train' for l in config['labels']] #used to identify proper labels to be used for training

    site = []
    target_variable = []
    weight_type = []
    weight_sum_across_nodes = []
    data, variables = preprocess(*site_zip_info, config['granularity'], config['target_variables'], config['backup_variables'], config['labels'], [], offset=config['offset'])
    print(variables)
    num_key_variables = len(variables) - 1 

    data[variables[:-1]] = scale(data[variables[:-1]])
    processed = data[variables + train_labels + config['labels']].astype('float64')

    # # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        processed[variables].to_numpy(), processed[train_labels].to_numpy(), test_size=config['test_size'], shuffle=False) #can't shuffle because time series data -> sequence order matters

    hyperparam_out = open(config['out'] + '/' + site_zip_info[1]+'_out.txt')
    best_params = None
    for line in hyperparam_out:
        if '{' in line:
            best_params = ast.literal_eval(line)
            break

    base_model = get_model_type(config['model_type'])

    if config['model_type'] == 'lstm':
        lstm = base_model(num_key_variables, 1)
        lstm.set_params(**best_params)
        # lstm.set_params(epochs=1)

        for i in range(num_iter):
            lstm.fit(X_train, y_train)

            param_list = list(lstm.model.parameters())
            dim = lstm.hidden_dim
            input_weights = param_list[0].data.cpu()
            ii_weights = torch.sum(input_weights[:dim], dim=0)
            if_weights = torch.sum(input_weights[dim:dim*2], dim=0)
            ig_weights = torch.sum(input_weights[dim*2:dim*3], dim=0)
            io_weights = torch.sum(input_weights[dim*3:dim*4], dim=0)

            for i in range(len(ii_weights)):
                site.append(site_zip_info[1])
                target_variable.append(variables[i])
                weight_type.append("input-input")
                weight_sum_across_nodes.append(ii_weights[i].item())

            for i in range(len(if_weights)):
                site.append(site_zip_info[1])
                target_variable.append(variables[i])
                weight_type.append("input-forget")
                weight_sum_across_nodes.append(if_weights[i].item())

            for i in range(len(ig_weights)):
                site.append(site_zip_info[1])
                target_variable.append(variables[i])
                weight_type.append("input-cell state")
                weight_sum_across_nodes.append(ig_weights[i].item())

            for i in range(len(io_weights)):
                site.append(site_zip_info[1])
                target_variable.append(variables[i])
                weight_type.append("input-output")
                weight_sum_across_nodes.append(io_weights[i].item())
    else:
        raise ValueError("weight variablity testing not supported for specified model type")


    df = pd.DataFrame({"site": site, "target_variable": target_variable, "weight_type": weight_type, "variability": weight_sum_across_nodes})
    path = config['out'] + '/weights'
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/' + site_zip_info[1] +'_weight_variance.txt', index=False)