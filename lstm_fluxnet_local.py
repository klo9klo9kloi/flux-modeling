from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, GridSearchCV
from processing_utils import preprocess, generate_visualizations, generate_file_output
from estimators import SimpleLSTMRegressor

def train_lstm_on_site(zip_info):
    #static variables across runs; initialize for each method call so that parallelization doesn't run into problems
    target_variables = ['TA_F', 'SW_IN_F', 'P_F', 'WS_F', 'VPD_F', 'CO2_F', 'time_index'] #time_index is a custom column added during preprocessing to aid in proper folding of training set
    backup_variables = {'TA_F' : 'TA_F_MDS', 'SW_IN_F': 'SW_IN_F_MDS', 'P_F': 'P_F_MDS', 'CO2_F': 'CO2_F_MDS', 'WS_F': 'WS_F_MDS', 'VPD_F': 'VPD_F_MDS'}
    labels = ['GPP_NT_VUT_REF']
    granularity = 'DD' #can be 'YY', 'WW', 'DD', 'MM', 'HH'
    test_size = 0.25
    k = 10

    file_output = []

    data = preprocess(*zip_info, granularity)

    # TODO: add some helpful print or output files to keep track of which variables are being used
    variables = []
    for v in target_variables:
        if v in data.columns:
            variables.append(v)
        elif backup_variables[v] in data.columns:
            variables.append(backup_variables[v])
            file_output.append('Using backup variable for ' + v)
        else:
            file_output.append('Variable ' + v + ' not in this dataset')

    num_key_variables = len(variables)-1 # dont count time_index

    # do any variable selection or dimensionality reduction here
    processed = data[variables + labels].iloc[0:100].astype('float32')

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        processed[variables].to_numpy(), processed[labels].to_numpy(), test_size=test_size, shuffle=False)

    # do grid search for best model
    tuned_parameters = [{'input_dim': [num_key_variables], 'output_dim': [1], 'hidden_dim': [256, 512, 1024], 'n_layers': [2,3,4,5], 'batch_size': [1, 10, 100], 
                        'seq_len': [5, 10, 15, 20, 25, 30], 'clip': [5, 10], 'epochs': [2, 3, 4, 5, 10], 'lr': [0.05, 0.005, 0.0005, 0.00005, 0.000005]}]
    # tuned_parameters = [{'input_dim': [num_key_variables], 'output_dim': [1], 'hidden_dim': [512], 'n_layers': [4], 'batch_size': [1], 
    #                     'seq_len': [5], 'clip': [5], 'epochs': [5], 'lr': [0.0005]}]

    clf = GridSearchCV(SimpleLSTMRegressor(), tuned_parameters, cv=k)
    clf.fit(X_train, y_train)

    # test best model
    file_output.append("Best parameters set found on for this site:")
    file_output.append(str(clf.best_params_))

    y_true, y_pred = y_test, clf.predict(X_test) #can call predict directly because refit=True
    train_true, train_pred = y_train, clf.predict(X_train)

    # visualize results
    generate_visualizations(np.append(train_true, y_true), train_pred, y_pred, clf.best_params_['seq_len'], granularity, data.iloc[0, 0], labels[0], tup[0])
    generate_file_output(file_output, tup[0])



