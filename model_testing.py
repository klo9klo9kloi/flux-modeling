from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from processing_utils import preprocess, fix_prediction_data, get_zip_info, generate_variability_graph, generate_generalizability_chart, generate_r2_chart
from estimators import SimpleLSTMRegressor, SimpleANNRegressor
import numpy as np
import pandas as pd
import os
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import operator

target_variables = ['TA_F', 'SW_IN_F', 'P_F', 'WS_F', 'VPD_F', 'CO2_F', 'SWC_F_MDS_1'] 
backup_variables = {'TA_F' : 'TA_F_MDS', 'SW_IN_F': 'SW_IN_F_MDS', 'P_F': 'P_F_MDS', 'CO2_F': 'CO2_F_MDS', 'WS_F': 'WS_F_MDS', 'VPD_F': 'VPD_F_MDS'}
labels = ['GPP_NT_VUT_REF']
train_labels = [l + '_train' for l in labels]

granularity = 'DD'
test_size = 0.25
val_size = 1.0/3.0
offset = False

def quantify_variability(zip_info, num_iter, regression_model):
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

    model = regression_model(num_key_variables, 1)
    model.set_params(**best_params)
    predictions = []
    scores = []
    for i in range(num_iter):
        print(i)
        if regression_model == SimpleLSTMRegressor:
            model.fit(X_train, y_train)
            test_pred = ([np.nan] * (model.sequence_length-1) ) + model.predict(X_test)
        elif regression_model == SimpleANNRegressor:
            model.fit(X_train[:, :-1], y_train)
            test_pred = model.predict(X_test[:, :-1])
        predictions.append(test_pred)
        prev_scoring = model.scoring
        model.set_params(scoring='r2')
        scores.append(model.score(X_test, y_test))
        # scores.append(model.score(X_test[:, :-1], y_test))
        model.set_params(scoring=prev_scoring)

    df = pd.DataFrame(data=np.array(predictions), columns=X_test[:, -1])
    # df[model.scoring] = scores
    df['r2'] = scores
    # path = 'out/predictions'
    path = '/content/gdrive/My Drive/Colab Notebooks/out/predictions'
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/' + zip_info[1] + '.txt', index=False)

    # generate_variability_graph(zip_info)

def test_generalizability(fluxnet_site_type, num_iter, regression_model):
    zip_file_info_for_climate_sites = get_zip_info(fluxnet_site_type)
    print(zip_file_info_for_climate_sites)

    # preprocess data for each site and store so we dont have to do it s^2 times
    processed_site_data = {}
    for zf in zip_file_info_for_climate_sites:
        data, variables = preprocess(*zf, granularity, target_variables, backup_variables, labels, [], offset=offset)
        print(variables)
        num_key_variables = len(variables) - 1 

        data[variables[:-1]] = scale(data[variables[:-1]])
        processed = data[variables + train_labels + labels].astype('float64')
        processed_site_data[zf[1]] = (processed, variables)

    # test using each site as the reference site
    site_name = []
    trained_on = []
    r2_score = []
    for reference_site in zip_file_info_for_climate_sites:
        reference_data, reference_variables = processed_site_data[reference_site[1]]
        # split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            reference_data[reference_variables].to_numpy(), reference_data[train_labels].to_numpy(), test_size=test_size, shuffle=False) #can't shuffle because it is time series data -> sequence order matters

        hyperparam_out = open('out/' + reference_site[1]+'_out.txt')
        best_params = None
        for line in hyperparam_out:
            if '{' in line:
                best_params = ast.literal_eval(line)
                break

        model = regression_model(num_key_variables, 1)
        model.set_params(**best_params)

        model.set_params(epochs=100)
        for i in range(num_iter):
            model.fit(X_train, y_train)
            # model.fit(X_train[:, :-1], y_train)
            model.set_params(scoring='r2')

            site_name.append(reference_site[1])
            trained_on.append(reference_site[1])
            r2_score.append(model.score(X_test, y_test))

            for other_site_info in zip_file_info_for_climate_sites:
                if other_site_info[1] != reference_site[1]:
                    other_site_data, other_site_variables = processed_site_data[other_site_info[1]]
                    other_site_data[other_site_variables[:-1]] = scale(other_site_data[other_site_variables[:-1]])
                    for v in reference_variables:
                        if v not in other_site_variables:
                            other_site_data[v] = np.zeros(len(other_site_data.index))
                    other_site_data=other_site_data.iloc[:100]
                    site_name.append(other_site_info[1])
                    trained_on.append(reference_site[1])
                    r2_score.append(model.score(other_site_data[reference_variables].to_numpy(), other_site_data[labels].to_numpy()))
                

    df = pd.DataFrame({'site': site_name, 'trained_on': trained_on, 'r2': r2_score})
    path = 'out/' + fluxnet_site_type 
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/generalizability_test.txt', index=False)

def test_universiality(fluxnet_site_type, num_iter):
    zip_file_info_for_climate_sites = get_zip_info(fluxnet_site_type)
    print(zip_file_info_for_climate_sites)

    # preprocess data for each site and store so we dont have to do it s^2 times
    processed_site_data = {}
    common_variable_set = set(target_variables)
    for zf in zip_file_info_for_climate_sites:
        site_data, site_variables = preprocess(*zf, granularity, target_variables, backup_variables, labels, [], offset=offset)
        print(site_variables)

        site_data[site_variables[:-1]] = scale(site_data[site_variables[:-1]])
        processed = site_data[site_variables + train_labels + labels].astype('float64')
        processed_site_data[zf[1]] = processed

        common_variable_set = common_variable_set.intersection(set(site_variables))

    site_test_data = {}
    common_variables = list(common_variable_set) + ['avg_fAPAR_interpol', 'time_index']
    print("Common variables: " + str(common_variables))
    cumulative_train_data = np.zeros((1, len(common_variables)))
    cumulative_val_data = np.zeros((1, len(common_variables)))
    cumulative_train_labels = np.zeros((1,1))
    cumulative_val_labels = np.zeros((1,1))
    for site_name in processed_site_data:
        d = processed_site_data[site_name]
        site_train, site_test, site_y_train, site_y_test = train_test_split(
            d[common_variables].to_numpy(), d[train_labels].to_numpy(), test_size=test_size, shuffle=False)
        # set aside test data for later
        site_test_data[site_name] = (site_test, site_y_test)

        # create train-val split
        site_train, site_val, site_y_train, site_y_val = train_test_split(
            site_train, site_y_train, test_size=val_size, shuffle=False)

        cumulative_train_data = np.concatenate((cumulative_train_data, site_train))
        cumulative_train_labels = np.concatenate((cumulative_train_labels, site_y_train))
        cumulative_val_data = np.concatenate((cumulative_val_data, site_val))
        cumulative_val_labels = np.concatenate((cumulative_val_labels, site_y_val))

    cumulative_train_data = cumulative_train_data[1:]
    cumulative_train_labels = cumulative_train_labels[1:]
    cumulative_val_data = cumulative_val_data[1:]
    cumulative_val_labels = cumulative_val_labels[1:]

    all_training_data = np.concatenate( (cumulative_train_data, cumulative_val_data) )
    all_training_labels = np.concatenate( (cumulative_train_labels, cumulative_val_labels) )
    train = list(range(len(cumulative_train_data)))
    test = list(range(len(cumulative_train_data), len(all_training_data)))

    num_key_variables = len(common_variables) - 1

    tuned_parameters = [{'batch_size': [30, 90], 'seq_len': [5, 15, 30], 'lr': [0.05, 0.005, 0.0005, 0.00005], 'regularization_param': [1, 0.1, 0.001, 0.0001]}]
    # tuned_parameters = [{'batch_size': [90], 'seq_len': [15, 30]}]

    clf = GridSearchCV(SimpleLSTMRegressor(num_key_variables, 1, hidden_dim=num_key_variables, epochs=2500, clip=10), 
                        tuned_parameters, cv=[(train, test)], refit=False, n_jobs=-1)
    clf.fit(all_training_data, all_training_labels)

    best_params = clf.best_params_

    path = 'out/' + fluxnet_site_type
    with open(path + '/universiality_best_params.txt', 'w') as f:
        f.write("Common variable set: \n")
        f.write(str(common_variables) + '\n')
        f.write("Best parameters: \n")
        f.write(str(best_params) + '\n')
        f.close()

    site_name = []
    r2_score = []

    for i in range(num_iter):
        model = SimpleLSTMRegressor(num_key_variables, 1, clip=10, epochs=2500, hidden_dim=num_key_variables)
        model.set_params(**best_params)
        ## For LSTM
        model.fit(all_training_data, all_training_labels)
        model.set_params(scoring='r2')
        for sn in site_test_data:
            site_name.append(sn)
            reference_data, reference_labels = site_test_data[sn]
            r2_score.append(model.score(reference_data, reference_labels))

    df = pd.DataFrame({'site': site_name, 'r2': r2_score}) 
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/universiality_test.txt', index=False)

def quantify_weight_variability(zip_infos, num_iter):
    site = []
    target_variable = []
    weight_type = []
    weight_sum_across_nodes = []
    for zip_info in zip_infos:
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
                site.append(zf[1])
                target_variable.append(variables[i])
                weight_type.append("input-input")
                weight_sum_across_nodes.append(ii_weights[i].item())

            for i in range(len(if_weights)):
                site.append(zf[1])
                target_variable.append(variables[i])
                weight_type.append("input-forget")
                weight_sum_across_nodes.append(if_weights[i].item())

            for i in range(len(ig_weights)):
                site.append(zf[1])
                target_variable.append(variables[i])
                weight_type.append("input-cell state")
                weight_sum_across_nodes.append(ig_weights[i].item())

            for i in range(len(io_weights)):
                site.append(zf[1])
                target_variable.append(variables[i])
                weight_type.append("input-output")
                weight_sum_across_nodes.append(io_weights[i].item())


    df = pd.DataFrame({"site": site, "target_variable": target_variable, "weight_type": weight_type, "variability": weight_sum_across_nodes})
    path = 'out/weights'
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/weight_variance_across_sites.txt', index=False)

    # generate_weight_variability_graph(zip_info)



if __name__ == '__main__':
    test_universiality('WSA', 100)
    # test_generalizability('WSA', 5, SimpleLSTMRegressor)
    # generate_generalizability_chart('WSA', "ANN")
    # zip_file_info_for_preprocessing = get_zip_info("testnet")
    # quantify_weight_variability(zip_file_info_for_preprocessing,50)
    # fix_prediction_data(zip_file_info_for_preprocessing)
    # generate_r2_chart(zip_file_info_for_preprocessing)
    # print(zip_file_info_for_preprocessing)
    # for zf in zip_file_info_for_preprocessing:
        # quantify_variability(zf, 100, SimpleLSTMRegressor)
        # generate_variability_graph(zf) 





