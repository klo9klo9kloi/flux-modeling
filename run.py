import os
import sys
import torch
import random
import numpy as np
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
	"""Tests the universality of a model by training on all data 
	from all sites of the same type and testing on each individual site.

	Keyword arguments:
	path_to_config_file -- relative path to config file (str)
	num_iter -- number of test iterations (int)
	site_type -- FLUXNET site type (str)
	extras -- additional figure title text
	"""
    num_iter = int(num_iter)
    test_universality(path_to_config_file, site_type, num_iter)
    config = get_training_params(path_to_config_file)
    generate_universality_chart(site_type, config['out'], config['viz'], config['model_type'], extras)

def weight_variability(path_to_data_directory, path_to_config_file, parallel, num_iter, extras=""):
	"""Quantifies the variability of a model's learned weights.

	Keyword arguments:
	path_to_data_directory -- relative path to directory containing FLUXNET zip files (str)
	path_to_config_file -- relative path to config file (str)
	parallel -- whther to compute in parallel (boolean)
	num_iter -- number of test iterations (int)
	extras -- additional figure title text
	"""
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
	"""Tests the generalizability of a model by training on data 
	from one site and testing on all other sites of the same type.

	Keyword arguments:
	path_to_data_directory -- relative path to directory containing FLUXNET zip files (str)
	path_to_config_file -- relative path to config file (str)
	parallel -- whther to compute in parallel (boolean)
	num_iter -- number of test iterations (int)
	site_type -- FLUXNET site type (str)
	extras -- additional figure title text
	"""
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
	"""Evaluates the basic performance of a model on specified test sites.

	Keyword arguments:
	path_to_data_directory -- relative path to directory containing FLUXNET zip files (str)
	path_to_config_file -- relative path to config file (str)
	parallel -- whther to compute in parallel (boolean)
	num_iter -- number of test iterations (int)
	extras -- additional figure title text
	"""
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
	"""Tests the generalizability of a model by training on data 
	from one site and testing on all other sites of the same type.

	Keyword arguments:
	path_to_data_directory -- relative path to directory containing FLUXNET zip files (str)
	path_to_config_file -- relative path to config file (str)
	parallel -- whther to compute in parallel (boolean)
	"""
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
	assert(len(sys.argv) >= 4)
	seed_num, run_type, args = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
	random.seed(seed_num)
	np.random.seed(seed_num)
	torch.manual_seed(seed_num)

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
		print("unrecognized run type command")




