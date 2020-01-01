from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from processing_utils import preprocess, get_zip_info, get_training_params
from estimators import get_model_type
import numpy as np
import pandas as pd
import os
import ast
def test_universality(path_to_config_file, fluxnet_site_type, num_iter):
    config = get_training_params(path_to_config_file)
    train_labels = [l + '_train' for l in config['labels']] #used to identify proper labels to be used for training

    base_model = get_model_type(config['model_type'])

    zip_file_info_for_climate_sites = get_zip_info(fluxnet_site_type)
    print(zip_file_info_for_climate_sites)

    # preprocess data for each site and store so we dont have to do it s^2 times
    processed_site_data = {}
    common_variable_set = set(config['target_variables'])
    for zf in zip_file_info_for_climate_sites:
        site_data, site_variables = preprocess(*zf, config['granularity'], config['target_variables'], config['backup_variables'], config['labels'], [], offset=config['offset'])
        print(site_variables)

        site_data[site_variables[:-1]] = scale(site_data[site_variables[:-1]])
        processed = site_data[site_variables + train_labels + config['labels']].astype('float64')
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
            d[common_variables].to_numpy(), d[train_labels].to_numpy(), test_size=config['test_size'], shuffle=False)
        # set aside test data for later
        site_test_data[site_name] = (site_test, site_y_test)

        # create train-val split
        site_train, site_val, site_y_train, site_y_val = train_test_split(
            site_train, site_y_train, test_size=config['val_size'], shuffle=False)

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

    clf = GridSearchCV(base_model(num_key_variables, 1), 
                        config['hyperparameter_grid'], cv=[(train, test)], refit=False, n_jobs=-1)
    clf.fit(all_training_data, all_training_labels)

    best_params = clf.best_params_

    path = config['out'] + '/' + fluxnet_site_type
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
        print("Iteration " + str(i))
        model = base_model(num_key_variables, 1)
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