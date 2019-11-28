from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from processing_utils import preprocess, fix_prediction_data, get_zip_info, generate_variability_graph, generate_generalizability_chart, generate_r2_chart
from estimators import SimpleLSTMRegressor, SimpleANNRegressor
import numpy as np
import pandas as pd
import os
import ast
import seaborn as sns
import matplotlib.pyplot as plt

def quantify_variability(zip_info, num_iter, regression_model):
    target_variables = ['TA_F', 'SW_IN_F', 'P_F', 'WS_F', 'VPD_F', 'CO2_F', 'SWC_F_MDS_1'] 
    backup_variables = {'TA_F' : 'TA_F_MDS', 'SW_IN_F': 'SW_IN_F_MDS', 'P_F': 'P_F_MDS', 'CO2_F': 'CO2_F_MDS', 'WS_F': 'WS_F_MDS', 'VPD_F': 'VPD_F_MDS'}
    labels = ['GPP_NT_VUT_REF']
    train_labels = [l + '_train' for l in labels]

    granularity = 'DD'
    test_size = 0.25
    offset = False

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
    # model = SimpleLSTMRegressor(num_key_variables, 1, hidden_dim=num_key_variables, n_layers=1, lr=0.0005, batch_size=90, seq_len = 15, epochs = 2500, threshold=1e-5, 
    #                 clip = 5, scoring='mse', regularization_param=0.001)
    # model = SimpleANNRegressor(num_key_variables, 1, hidden_dim=num_key_variables, batch_size= 1, epochs=10, lr= 0.0005, regularization_param= 0.1)
    predictions = []
    scores = []
    for i in range(num_iter):
        # model.fit(X_train, y_train)
        model.fit(X_train[:, :-1], y_train)
        # test_pred = ([np.nan] * (model.sequence_length-1) ) + model.predict(X_test)
        test_pred = model.predict(X_test[:, :-1])
        predictions.append(test_pred)
        prev_scoring = model.scoring
        model.set_params(scoring='r2')
        # scores.append(model.score(X_test, y_test))
        scores.append(model.score(X_test[:, :-1], y_test))
        model.set_params(scoring=prev_scoring)

    df = pd.DataFrame(data=np.array(predictions), columns=X_test[:, -1])
    # df[model.scoring] = scores
    df['r2'] = scores
    path = 'out/predictions'
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/' + zip_info[1] + '.txt', index=False)

    # generate_variability_graph(zip_info)

def test_generalizability(fluxnet_site_type, num_iter, regression_model):
    zip_file_info_for_climate_sites = get_zip_info(fluxnet_site_type)
    print(zip_file_info_for_climate_sites)

    target_variables = ['TA_F', 'SW_IN_F', 'P_F', 'WS_F', 'VPD_F', 'CO2_F', 'SWC_F_MDS_1'] 
    backup_variables = {'TA_F' : 'TA_F_MDS', 'SW_IN_F': 'SW_IN_F_MDS', 'P_F': 'P_F_MDS', 'CO2_F': 'CO2_F_MDS', 'WS_F': 'WS_F_MDS', 'VPD_F': 'VPD_F_MDS'}
    labels = ['GPP_NT_VUT_REF']
    train_labels = [l + '_train' for l in labels]

    granularity = 'DD'
    test_size = 0.25
    offset = False

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

if __name__ == '__main__':
    # test_generalizability('WSA', 5, SimpleLSTMRegressor)
    # generate_generalizability_chart('WSA', "LSTM")
    zip_file_info_for_preprocessing = get_zip_info("testnet")
    # fix_prediction_data(zip_file_info_for_preprocessing)
    # generate_r2_chart(zip_file_info_for_preprocessing)
    # print(zip_file_info_for_preprocessing)
    for zf in zip_file_info_for_preprocessing:
        # quantify_variability(zf, 100, SimpleANNRegressor)
        generate_variability_graph(zf) 





