from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, GridSearchCV
from processing_utils import preprocess, generate_visualizations, generate_file_output, validate_variable, generate_weights_visualization
from estimators import SimpleLSTMRegressor
import numpy as np

def train_lstm_on_site(zip_info):
    #static variables across runs; initialize for each method call so that parallelization doesn't run into problems
    target_variables = ['TA_F', 'SW_IN_F', 'P_F', 'WS_F', 'VPD_F', 'time_index'] #time_index is a custom column added during preprocessing to aid in proper folding of training set
    backup_variables = {'TA_F' : 'TA_F_MDS', 'SW_IN_F': 'SW_IN_F_MDS', 'P_F': 'P_F_MDS', 'CO2_F': 'CO2_F_MDS', 'WS_F': 'WS_F_MDS', 'VPD_F': 'VPD_F_MDS'}
    labels = ['GPP_NT_VUT_REF']
    train_labels = [l + '_train' for l in labels]

    granularity = 'DD' #can be 'YY', 'WW', 'DD', 'MM', 'HH'
    test_size = 0.25
    k = 5

    file_output = []
    data = preprocess(*zip_info, granularity, labels)

    variables = []
    for v in target_variables:
        if validate_variable(data, v):
            variables.append(v)
        elif validate_variable(data, backup_variables[v]):
            variables.append(backup_variables[v])
            file_output.append('Using backup variable for ' + v)
        else:
            file_output.append('Variable ' + v + ' is not used during training for this dataset, either because it is missing or has missing values.')

    num_key_variables = len(variables)-1 # dont count time_index

    # do any variable selection or dimensionality reduction here
    processed = data[variables + train_labels + labels].astype('float64')
    # print(processed.describe())

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        processed[variables].to_numpy(), processed[train_labels].to_numpy(), test_size=test_size, shuffle=False) #can't shuffle because it is time series data -> sequence order matters


    # do grid search for best model
    tuned_parameters = [{'hidden_dim': [256], 'batch_size': [30], 'seq_len': [15], 'lr': [0.005, 0.0005, 0.00005, 0.000005]}]
    # tuned_parameters = [{'hidden_dim': [50], 'batch_size': [10], 'seq_len': [5], 'clip': [5], 'epochs': [10], 'lr': [0.005]}]

    clf = GridSearchCV(SimpleLSTMRegressor(num_key_variables, 1), tuned_parameters, cv=k)
    clf.fit(X_train, y_train)

    # test best model
    best_model = clf.best_estimator_
    file_output.append("Best parameters set found on for this site:")
    file_output.append(str(clf.best_params_))
    file_output.append("Model score on test set with best parameters (" +  clf.best_estimator_.scoring + "):")
    file_output.append(str(best_model.score(X_test, y_test)))
    file_output.append("R2 score on test set with best parameters:")
    file_output.append(str(best_model.r2_score(X_test, y_test)))

    y_true, y_pred = y_test, best_model.predict(X_test) #can call predict directly because refit=True
    train_true, train_pred = y_train, best_model.predict(X_train)

    # visualize results
    generate_visualizations(processed[labels].to_numpy().squeeze(), train_pred, y_pred, clf.best_params_['seq_len'], granularity, data.iloc[0, 0], labels[0], zip_info[0])
    generate_file_output(file_output, zip_info[0])

    generate_weights_visualization(best_model, variables[:-1], zip_info[0])

    # generate_cell_state_visualization(clf.best_estimator_, processed[variables].to_numpy())



