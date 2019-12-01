from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from processing_utils import preprocess, get_zip_info
from estimators import SimpleLSTMRegressor, SimpleANNRegressor
import numpy as np
import pandas as pd
import os
import ast
def test_universality(fluxnet_site_type, num_iter):
    target_variables = ['TA_F', 'SW_IN_F', 'P_F', 'WS_F', 'VPD_F', 'CO2_F_MDS', 'SWC_F_MDS_1'] 
    backup_variables = {'TA_F' : 'TA_F_MDS', 'SW_IN_F': 'SW_IN_F_MDS', 'P_F': 'P_F_MDS', 'CO2_F': 'CO2_F_MDS', 'WS_F': 'WS_F_MDS', 'VPD_F': 'VPD_F_MDS'}
    labels = ['GPP_NT_VUT_REF']
    train_labels = [l + '_train' for l in labels]

    granularity = 'DD'
    test_size = 0.25
    val_size = 1.0/3.0
    offset = True

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
    if not os.path.exists(path):
        os.makedirs(path)

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
    df.to_csv(path + '/universiality_test.txt', index=False)