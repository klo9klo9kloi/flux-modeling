from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from processing_utils import preprocess
from estimators import SimpleLSTMRegressor, SimpleANNRegressor
import numpy as np
import pandas as pd
import os
import ast
import torch
def quantify_weight_variability(zip_info, num_iter=100):
    target_variables = ['TA_F', 'SW_IN_F', 'P_F', 'WS_F', 'VPD_F', 'CO2_F', 'SWC_F_MDS_1'] 
    backup_variables = {'TA_F' : 'TA_F_MDS', 'SW_IN_F': 'SW_IN_F_MDS', 'P_F': 'P_F_MDS', 'CO2_F': 'CO2_F_MDS', 'WS_F': 'WS_F_MDS', 'VPD_F': 'VPD_F_MDS'}
    labels = ['GPP_NT_VUT_REF']
    train_labels = [l + '_train' for l in labels]

    granularity = 'DD'
    test_size = 0.25
    offset = True

    site = []
    target_variable = []
    weight_type = []
    weight_sum_across_nodes = []
    data, variables = preprocess(*zip_info, granularity, target_variables, backup_variables, labels, [], offset=offset)
    print(variables)
    num_key_variables = len(variables) - 1 

    data[variables[:-1]] = scale(data[variables[:-1]])
    processed = data[variables + train_labels + labels].astype('float64')

    # # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        processed[variables].to_numpy(), processed[train_labels].to_numpy(), test_size=test_size, shuffle=False) #can't shuffle because it is time series data -> sequence order matters

    hyperparam_out = open('out/' + zip_info[1]+'_out.txt')
    best_params = None
    for line in hyperparam_out:
        if '{' in line:
            best_params = ast.literal_eval(line)
            break

    lstm = SimpleLSTMRegressor(num_key_variables, 1)
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
            site.append(zip_info[1])
            target_variable.append(variables[i])
            weight_type.append("input-input")
            weight_sum_across_nodes.append(ii_weights[i].item())

        for i in range(len(if_weights)):
            site.append(zip_info[1])
            target_variable.append(variables[i])
            weight_type.append("input-forget")
            weight_sum_across_nodes.append(if_weights[i].item())

        for i in range(len(ig_weights)):
            site.append(zip_info[1])
            target_variable.append(variables[i])
            weight_type.append("input-cell state")
            weight_sum_across_nodes.append(ig_weights[i].item())

        for i in range(len(io_weights)):
            site.append(zip_info[1])
            target_variable.append(variables[i])
            weight_type.append("input-output")
            weight_sum_across_nodes.append(io_weights[i].item())


    df = pd.DataFrame({"site": site, "target_variable": target_variable, "weight_type": weight_type, "variability": weight_sum_across_nodes})
    path = 'out/weights'
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/' + zip_info[1] +'_weight_variance.txt', index=False)

    # generate_weight_variability_graph(zip_info)