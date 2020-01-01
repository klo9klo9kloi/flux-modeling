from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from processing_utils import preprocess, get_training_params
from estimators import get_model_type
import numpy as np
import pandas as pd
import os
import ast

def test_performance(site_zip_info, path_to_config_file, num_iter):
    config = get_training_params(path_to_config_file)
    train_labels = [l + '_train' for l in config['labels']] #used to identify proper labels to be used for training

    data, variables = preprocess(*site_zip_info, config['granularity'], config['target_variables'], config['backup_variables'], config['labels'], [], offset=config['offset'])
    print("Training Variables for " + site_zip_info[1] + ":")
    print(variables)
    num_key_variables = len(variables) - 1 #dont count time index; time index is only used for custom sampler

    data[variables[:-1]] = scale(data[variables[:-1]])
    processed = data[variables + train_labels + config['labels']].astype('float64')

    # # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        processed[variables].to_numpy(), processed[train_labels].to_numpy(), test_size=config['test_size'], shuffle=False) #can't shuffle because time series data -> sequence order matters
    hyperparam_out = open(config['out'] + '/' + site_zip_info[1]+'_out.txt')
    best_params = None
    for line in hyperparam_out:
        if '{' in line:
            best_params = ast.literal_eval(line)
            break

    base_model = get_model_type(config['model_type'])

    model = base_model(num_key_variables, 1)
    model.set_params(**best_params)
    predictions = []
    scores = []
    for i in range(num_iter):
        print("Iteration " + str(i))
        if config['model_type'] == 'lstm':
            model.fit(X_train, y_train)
            test_pred = ([np.nan] * (model.sequence_length-1) ) + model.predict(X_test)
        elif config['model_type'] == 'ann':
            model.fit(X_train[:, :-1], y_train)
            test_pred = model.predict(X_test[:, :-1])
        predictions.append(test_pred)
        prev_scoring = model.scoring
        model.set_params(scoring='r2')
        if config['model_type'] == 'lstm':
            scores.append(model.score(X_test, y_test))
        elif config['model_type'] == 'ann':
            scores.append(model.score(X_test[:, :-1], y_test))
        model.set_params(scoring=prev_scoring)

    df = pd.DataFrame(data=np.array(predictions), columns=X_test[:, -1])
    df['r2'] = scores
    path = config['out'] + '/predictions'
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path + '/' + site_zip_info[1] + '.txt', index=False)