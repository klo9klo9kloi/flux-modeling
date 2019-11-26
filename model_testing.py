from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from processing_utils import preprocess, generate_visualizations, generate_file_output, generate_weights_visualization, get_zip_info
from estimators import SimpleLSTMRegressor, SimpleANNRegressor
import numpy as np
import pandas as pd
import os
import ast

def quantify_variability(zip_info):
    target_variables = ['TA_F', 'SW_IN_F', 'P_F', 'WS_F', 'VPD_F', 'CO2_F', 'SWC_F_MDS_1'] 
    backup_variables = {'TA_F' : 'TA_F_MDS', 'SW_IN_F': 'SW_IN_F_MDS', 'P_F': 'P_F_MDS', 'CO2_F': 'CO2_F_MDS', 'WS_F': 'WS_F_MDS', 'VPD_F': 'VPD_F_MDS'}
    labels = ['GPP_NT_VUT_REF']
    train_labels = [l + '_train' for l in labels]

    granularity = 'DD'
    test_size = 0.25
    offset = False

    file_output = []
    data, variables = preprocess(*zip_info, granularity, target_variables, backup_variables, labels, file_output, offset=offset)
    print(variables)
    num_key_variables = len(variables) - 1 

    data[variables[:-1]] = scale(data[variables[:-1]])
    processed = data[variables + train_labels + labels].iloc[:100].astype('float64')

    # # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        processed[variables].to_numpy(), processed[train_labels].to_numpy(), test_size=test_size, shuffle=False) #can't shuffle because it is time series data -> sequence order matters

    hyperparam_out = open('out/' + zip_info[0]+'_out.txt')
    best_params = None
    for line in hyperparam_out:
        if '{' in line:
            best_params = ast.literal_eval(line)
            break

    model = SimpleLSTMRegressor(num_key_variables, 1)
    model.set_params(**best_params)
    # model = SimpleLSTMRegressor(num_key_variables, 1, hidden_dim=num_key_variables, n_layers=1, lr=0.0005, batch_size=90, seq_len = 15, epochs = 2500, threshold=1e-5, 
    #                 clip = 5, scoring='mse', regularization_param=0.001)
    # model = SimpleANNRegressor(num_key_variables, 1, hidden_dim=num_key_variables, batch_size= 1, epochs=10, lr= 0.0005, regularization_param= 0.1)
    predictions = []
    for i in range(5):
        model.fit(X_train, y_train)
        # model.fit(X_train[:, :-1], y_train)
        y_test = ([np.nan] * (model.sequence_length-1) ) + model.predict(X_test)
        # y_test = model.predict(X_test[:, :-1])
        predictions.append(y_test)
    df = pd.DataFrame(data=np.array(predictions), columns=X_test[:, -1])
    path = 'out/predictions'
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/' + zip_info[0] + '.txt', index=False)

if __name__ == '__main__':
    zip_file_info_for_preprocessing = get_zip_info()
    print(zip_file_info_for_preprocessing)
    # with Pool(5) as p:
    #   p.map(train_lstm_on_site, zip_file_info_for_preprocessing)
    for zf in zip_file_info_for_preprocessing:
        quantify_variability(zf)





