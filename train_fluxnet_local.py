from sklearn.preprocessing import normalize, scale
from sklearn.model_selection import train_test_split, GridSearchCV
from processing_utils import get_training_params, preprocess, generate_visualizations, generate_file_output, generate_weights_visualization
from estimators import get_model_type
import numpy as np

def train_on_site(site_zip_info, path_to_config_file):
    """Train a model with the given configurations using data from the specified site.
    General pipeline:
        - preprocess data 
        - hyperparameter search
        - scoring
        - graping

    Keyword arguments:
    site_zip_info -- zipfile information about specified site (tup)
    path_to_config_file -- relative path to config file (str)
    """
    config = get_training_params(path_to_config_file)
    train_labels = [l + '_train' for l in config['labels']] #used to identify proper labels to be used for training

    file_output = []
    data, variables = preprocess(*site_zip_info, config['granularity'], config['target_variables'], config['backup_variables'], config['labels'], file_output, offset=config['offset'])
    print("Training Variables for " + site_zip_info[1] + ":")
    print(variables)
    num_key_variables = len(variables) - 1 #dont count time index; time index is only used for custom sampler

    data[variables[:-1]] = scale(data[variables[:-1]])
    processed = data[variables + train_labels + config['labels']].astype('float64')

    base_model = get_model_type(config['model_type'])

    # # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        processed[variables].to_numpy(), processed[train_labels].to_numpy(), test_size=config['test_size'], shuffle=False) #don't shuffle time series data -> sequence order matters

    clf = GridSearchCV(base_model(num_key_variables, 1), config['hyperparameter_grid'], cv=config['k'])
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
    y_train = best_model.predict(X_train)
    y_test = best_model.predict(X_test)
    if config['model_type'] == 'lstm':
        if config['offset'] > 0:
            y_train = ([np.nan] * (best_model.sequence_length + config['offset'] - 1 )) + y_train[:-config['offset']]
            y_test = ([np.nan] * (best_model.sequence_length + config['offset'] - 1 )) + y_test[:-config['offset']]
        else:
            y_train = ([np.nan] * (best_model.sequence_length - 1 )) + y_train
            y_test = ([np.nan] * (best_model.sequence_length - 1 )) + y_test

    generate_visualizations(processed['time_index'].to_numpy().squeeze(), processed[config['labels']].to_numpy().squeeze(), 
                                y_test, y_train, config['granularity'], data['TIMESTAMP'].iloc[0], config['labels'], 
                                site_zip_info[1], config['viz'])
    generate_file_output(file_output, site_zip_info[1], config['out'])

    if config['model_type'] == 'lstm':
        generate_weights_visualization(best_model, variables[:-1], site_zip_info[1], config['viz'])




