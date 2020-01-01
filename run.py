import os
import sys
from processing_utils import get_training_params, get_zip_info, generate_variability_graph, generate_generalizability_chart, generate_r2_chart, generate_weight_variance_chart, generate_universality_chart
from train_fluxnet_local import train_on_site
from functools import partial
from test_generalizability import test_generalizability
from test_universality import test_universality
from quantify_weight_variability import quantify_weight_variability
from test_performance import test_performance
from multiprocessing import Pool, Manager

parallel_true_values = ["True", "true", "T", "t"]

def universality(path_to_config_file, num_iter, site_type, extras=""):
    num_iter = int(num_iter)
    test_universality(path_to_config_file, site_type, num_iter)
    config = get_training_params(path_to_config_file)
    generate_universality_chart(site_type, config['out'], config['viz'], config['model_type'], extras)

def weight_variability(path_to_data_directory, path_to_config_file, parallel, num_iter, extras=""):
    num_iter = int(num_iter)
    parallel = (parallel in parallel_true_values)
    zip_file_info_for_preprocessing = get_zip_info(path_to_data_directory)
    print(zip_file_info_for_preprocessing)
    if parallel:
        with Pool() as p:
           p.map(partial(quantify_weight_variability, path_to_config_file=path_to_config_file, num_iter=num_iter), zip_file_info_for_preprocessing)
    else:
        for zf in zip_file_info_for_preprocessing:
            quantify_weight_variability(zf, path_to_config_file, num_iter)
    config = get_training_params(path_to_config_file)
    generate_weight_variance_chart(zip_file_info_for_preprocessing, config['out'], config['viz'], extras)


def generalizability(path_to_data_directory, path_to_config_file, parallel, num_iter, site_type, extras=""):
    num_iter = int(num_iter)
    parallel = (parallel in parallel_true_values)
    zip_file_info_for_preprocessing = get_zip_info(path_to_data_directory)
    print(zip_file_info_for_preprocessing)
    if parallel:
        with Pool() as p:
           p.map(partial(test_generalizability, path_to_config_file=path_to_config_file, num_iter=num_iter, fluxnet_site_type=site_type), zip_file_info_for_preprocessing)
    else:
        for zf in zip_file_info_for_preprocessing:
            test_generalizability(zf, path_to_config_file, site_type, num_iter)
    config = get_training_params(path_to_config_file)
    generate_generalizability_chart(site_type, config['out'], config['viz'], config['model_type'], extras)

def performance(path_to_data_directory, path_to_config_file, parallel, num_iter, extras=""):
    num_iter = int(num_iter)
    parallel = (parallel in parallel_true_values)
    zip_file_info_for_preprocessing = get_zip_info(path_to_data_directory)
    print(zip_file_info_for_preprocessing)
    if parallel:
        with Pool() as p:
           p.map(partial(test_performance, path_to_config_file=path_to_config_file, num_iter=num_iter), zip_file_info_for_preprocessing)
    else:
        for zf in zip_file_info_for_preprocessing:
            test_performance(zf, path_to_config_file, num_iter)
    config = get_training_params(path_to_config_file)
    for zf in zip_file_info_for_preprocessing:
        generate_variability_graph(zf, config['out'], config['viz'], config['model_type'], extras) 
    generate_r2_chart(zip_file_info_for_preprocessing, config['out'], config['viz'], config['model_type'], extras)

def train(path_to_data_directory, path_to_config_file, parallel):
    zip_file_info_for_preprocessing = get_zip_info(path_to_data_directory)
    parallel = (parallel in parallel_true_values)
    print(zip_file_info_for_preprocessing)

    if parallel:
        with Pool() as p:
            p.map(partial(train_on_site, path_to_config_file=path_to_config_file), zip_file_info_for_preprocessing)
    else:
        for zf in zip_file_info_for_preprocessing:
            train_on_site(zf, path_to_config_file)


if __name__ == '__main__':
    run_type, args = sys.argv[1], sys.argv[2:]
    if run_type == 'train':
        train(*args)
    elif run_type == 'test':
    	performance(*args)
    elif run_type == "gen":
        generalizability(*args)
    elif run_type == 'uni':
        universality(*args)
    elif run_type == 'weight_viz':
    	weight_variability(*args)
    else:
        print("unrecognized command")




