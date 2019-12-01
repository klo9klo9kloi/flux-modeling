from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from processing_utils import preprocess, get_zip_info
from estimators import SimpleLSTMRegressor, SimpleANNRegressor
import numpy as np
import pandas as pd
import os
import ast

def test_generalizability(zip_info, fluxnet_site_type, num_iter):
    target_variables = ['TA_F', 'SW_IN_F', 'P_F', 'WS_F', 'VPD_F', 'CO2_F', 'SWC_F_MDS_1'] 
    backup_variables = {'TA_F' : 'TA_F_MDS', 'SW_IN_F': 'SW_IN_F_MDS', 'P_F': 'P_F_MDS', 'CO2_F': 'CO2_F_MDS', 'WS_F': 'WS_F_MDS', 'VPD_F': 'VPD_F_MDS'}
    labels = ['GPP_NT_VUT_REF']
    train_labels = [l + '_train' for l in labels]

    granularity = 'DD'
    test_size = 0.25
    offset = True

    zip_file_info_for_climate_sites = get_zip_info(fluxnet_site_type)
    print(zip_file_info_for_climate_sites)

    # preprocess data for each site and store so we dont have to do it s^2 times
    processed_site_data = {}
    X_train, X_test, y_train, y_test = None, None, None, None
    for zf in zip_file_info_for_climate_sites:
        data, variables = preprocess(*zf, granularity, target_variables, backup_variables, labels, [], offset=offset)

        data[variables[:-1]] = scale(data[variables[:-1]])
        processed = data[variables + train_labels + labels].astype('float64')
        processed_site_data[zf[1]] = (processed, variables)

        if zf[1] == zip_info[1]:
            reference_data, reference_variables = data, variables
            print(reference_variables)
            num_key_variables = len(reference_variables) - 1

            # split the dataset
            X_train, X_test, y_train, y_test = train_test_split(
                reference_data[reference_variables].to_numpy(), reference_data[train_labels].to_numpy(), test_size=test_size, shuffle=False) #can't shuffle because it is time series data -> sequence order matters
    
    

    hyperparam_out = open('out/' + zip_info[1]+'_out.txt', 'r')
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
        print(zip_info[1] + " iteration: " + str(i))
        model = SimpleLSTMRegressor(num_key_variables, 1)
        if best_params is not None:
            model.set_params(**best_params)
        model.set_params(epochs=1)
        model.fit(X_train, y_train)
        model.set_params(scoring='r2')

        for other_site_name in processed_site_data:
            if other_site_name != zip_info[1]:
                other_site_data, other_site_variables = processed_site_data[other_site_name]

                for v in reference_variables:
                    if v not in other_site_variables:
                        other_site_data[v] = np.zeros(len(other_site_data.index))
                print("Trained on: " + zip_info[1] + ", Scoring on: " + other_site_name)
                site_name.append(other_site_name)
                trained_on.append(zip_info[1])
                r2_score.append(model.score(other_site_data[reference_variables].to_numpy(), other_site_data[train_labels].to_numpy()))
                    

    df = pd.DataFrame({'site': site_name, 'trained_on': trained_on, 'r2': r2_score})
    path = 'out/' + fluxnet_site_type 
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/' + zip_info[1] + '_generalizability_test.txt', index=False)
