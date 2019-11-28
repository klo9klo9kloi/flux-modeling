from sklearn.preprocessing import normalize, scale
from sklearn.model_selection import train_test_split, GridSearchCV
from processing_utils import preprocess, generate_visualizations, generate_file_output, generate_weights_visualization
from estimators import SimpleANNRegressor
import numpy as np

def train_ann_on_site(zip_info):
    #static variables across runs; initialize for each method call so that parallelization doesn't run into problems
    target_variables = ['TA_F', 'SW_IN_F', 'P_F', 'WS_F', 'VPD_F', 'CO2_F', 'SWC_F_MDS_1'] 
    backup_variables = {'TA_F' : 'TA_F_MDS', 'SW_IN_F': 'SW_IN_F_MDS', 'P_F': 'P_F_MDS', 'CO2_F': 'CO2_F_MDS', 'WS_F': 'WS_F_MDS', 'VPD_F': 'VPD_F_MDS'}
    labels = ['GPP_NT_VUT_REF']
    train_labels = [l + '_train' for l in labels]

    granularity = 'DD' #can be 'YY', 'WW', 'DD', 'MM', 'HH'
    test_size = 0.25
    k = 5

    file_output = []
    data, variables = preprocess(*zip_info, granularity, target_variables, backup_variables, labels, file_output)
    print(variables) 
    variables = variables[:-1]
    num_key_variables = len(variables)

    # do any variable selection or dimensionality reduction here
    data[variables] = scale(data[variables])
    processed = data[variables + train_labels + labels + ['time_index']].astype('float64')
    # print(processed[variables])
    # print(variables)

    # # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        processed[variables].to_numpy(), processed[train_labels].to_numpy(), test_size=test_size, shuffle=False)
    # do grid search for best model
    # tuned_parameters = [{'hidden_dim': [num_key_variables], 'batch_size': [30, 90],'lr': [0.05, 0.005, 0.0005, 0.00005], 
    #                     'epochs': [2500], 'regularization_param': [1, 0.1, 0.01, 0.001]}]
    tuned_parameters = [{'hidden_dim': [num_key_variables], 'batch_size': [90],'lr': [0.0005], 
                        'epochs': [100], 'regularization_param': [0.1]}]

    clf = GridSearchCV(SimpleANNRegressor(num_key_variables, 1), tuned_parameters, cv=k)
    clf.fit(X_train, y_train)

    # test best model
    best_model = clf.best_estimator_
    file_output.append("Number of epochs trained for on this site:")
    file_output.append(str(best_model.trained_for + 1))
    file_output.append("Best parameters set found on for this site:")
    file_output.append(str(clf.best_params_))
    file_output.append("Model score on test set with best parameters (" +  clf.best_estimator_.scoring + "):")
    file_output.append(str(best_model.score(X_test, y_test)))
    best_model.set_params(scoring='r2')
    file_output.append("R2 score on train set with best parameters:")
    file_output.append(str(best_model.score(X_train, y_train)))
    file_output.append("R2 score on test set with best parameters:")
    file_output.append(str(best_model.score(X_test, y_test)))

    # visualize results
    y_test = best_model.predict(X_test)
    y_train = best_model.predict(X_train)
    generate_visualizations(processed['time_index'].to_numpy().squeeze(), processed[labels].to_numpy().squeeze(), y_test, y_train, 0, granularity, data['TIMESTAMP'].iloc[0], labels, zip_info[1])
    generate_file_output(file_output, zip_info[1])

    # generate_weights_visualization(best_model, variables[:-1], zip_info[0])




