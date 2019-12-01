from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from processing_utils import preprocess
from estimators import SimpleLSTMRegressor, SimpleANNRegressor
import numpy as np
import pandas as pd
import os
import ast

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
    path = 'out/predictions'
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/' + zip_info[1] + '.txt', index=False)