from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from processing_utils import preprocess, get_zip_info, get_training_params
from estimators import get_model_type
import numpy as np
import pandas as pd
import os
import ast

def test_generalizability(site_zip_info, path_to_config_file, fluxnet_site_type, num_iter):
    config = get_training_params(path_to_config_file)

    train_labels = [l + '_train' for l in config['labels']] #used to identify proper labels to be used for training
    base_model = get_model_type(config['model_type'])

    zip_file_info_for_climate_sites = get_zip_info(fluxnet_site_type)
    print(zip_file_info_for_climate_sites)

    # preprocess data for each site and store so we dont have to do it s^2 times
    processed_site_data = {}
    X_train, X_test, y_train, y_test = None, None, None, None
    found = False
    for zf in zip_file_info_for_climate_sites:
        data, variables = preprocess(*zf, config['granularity'], config['target_variables'], config['backup_variables'], config['labels'], [], offset=config['offset'])

        data[variables[:-1]] = scale(data[variables[:-1]])
        processed = data[variables + train_labels + config['labels']].astype('float64')
        processed_site_data[zf[1]] = (processed, variables)

        if zf[1] == site_zip_info[1]:
            found = True
            reference_data, reference_variables = data, variables
            print("Training Variables for evaluating " + site_zip_info[1] + ":")
            print(reference_variables)
            num_key_variables = len(reference_variables) - 1

            # split the dataset
            X_train, X_test, y_train, y_test = train_test_split(
                reference_data[reference_variables].to_numpy(), reference_data[train_labels].to_numpy(), test_size=config['test_size'], shuffle=False) #can't shuffle because time series data -> sequence order matters

    if not found:
        raise RuntimeError("Reference site not found within data directory")


    hyperparam_out = open(config['out'] + '/' + site_zip_info[1]+'_out.txt', 'r')
    best_params = None
    for line in hyperparam_out:
        if '{' in line:
            best_params = ast.literal_eval(line)
            break

    # test using each site as the reference site
    site_name = []
    trained_on = []
    r2_score = []
    for i in range(num_iter):
        print(site_zip_info[1] + " iteration: " + str(i))
        model = base_model(num_key_variables, 1)
        if best_params is not None:
            model.set_params(**best_params)
        model.fit(X_train, y_train)
        model.set_params(scoring='r2')

        for other_site_name in processed_site_data:
            if other_site_name != site_zip_info[1]:
                other_site_data, other_site_variables = processed_site_data[other_site_name]

                for v in reference_variables:
                    if v not in other_site_variables:
                        other_site_data[v] = np.zeros(len(other_site_data.index))
                print("Trained on: " + site_zip_info[1] + ", Scoring on: " + other_site_name)
                site_name.append(other_site_name)
                trained_on.append(site_zip_info[1])
                r2_score.append(model.score(other_site_data[reference_variables].to_numpy(), other_site_data[train_labels].to_numpy()))
                    

    df = pd.DataFrame({'site': site_name, 'trained_on': trained_on, 'r2': r2_score})
    path = config['out'] + '/' + fluxnet_site_type 
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/' + site_zip_info[1] + '_generalizability_test.txt', index=False)
