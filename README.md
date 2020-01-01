## flux-modeling
Keenan Group ML Exploration

The data used by this code can be downloaded from here: https://fluxnet.fluxdata.org/login/?redirect_to=/data/download-data/. 
Remote sensing data can be downloaded from here: 

# Dependencies
python==3.6.5  
pytorch==1.0.1.post2  
matplotlib==2.2.2  
numpy==1.14.3  
seaborn==0.9.0  
sklearn==0.19.1  

# Setup
The code can be run directly using the commands described below once the data has been downloaded and placed in the same working directory. Note that paths to all data and output directories specified by config files should also point to a subdirectory of the current working directory.

# Config Files
These text files are used to specify the configurations of a desired training pipeline/procedure for an experiment. They adhere to the following structure:

- model_type (ann or lstm)
- hyperparameter_grid (a list of dictionaries whose keys correspond to model parameters and whose values are a list of possible values)
- num_folds (the number of folds during cross validation)
- target_variables (a list of FLUXNET2015 variable names)
- backup_variables (a dictionary whose keys correspond to a target variable and whose values are a backup FLUXNET2015 variable)
- labels (a list of FLUXNET2015 variable names)
- granularity (YY, MM, WW, DD, or HH)
- val_size (a float between 0 and 1 non-inclusive specifying the size of the validation set)
- test_size (^ but for the test set)
- offset (a positive integer that specifies the number of days ahead to forecast)
- num_iter (the number of test iterations; only used for testing)
- out_dir (path to the desired output directory; will be created if does not already exist)
- viz_dir (path to the desired visualization output directory)

# Running Experiments
To train the model:  
`python3 run.py train path_to_data_directory path_to_config_file parallel`  

To evaluate the model via R^2:  
`python3 run.py test path_to_data_directory path_to_config_file parallel num_iter extras`  

To test generalizability of a model trained on one site to other similar sites:  
`python3 run.py gen path_to_data_directory path_to_config_file parallel num_iter site_type extras`  

To visualize the weight variance of the model parameters:    
`python3 run.py weight_viz path_to_data_directory path_to_config_file parallel num_iter extras`  

To test performance of a model trained on all similar sites at each individual site:    
`python3 run.py uni path_to_config_file num_iter site_type extras`  
