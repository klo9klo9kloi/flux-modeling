from processing_utils import get_zip_info, generate_variability_graph, generate_generalizability_chart, generate_r2_chart
import os
from functools import partial
from test_generalizability import test_generalizability
from test_universality import test_universality
from quantify_weight_variability import quantify_weight_variability
from multiprocessing import Pool, Manager

if __name__ == '__main__':
    # test_universality('WSA', 100)
    
    # generate_generalizability_chart('WSA', "ANN")
    zip_file_info_for_preprocessing = get_zip_info("testnet")
    # quantify_weight_variability(zip_file_info_for_preprocessing,50)
    # fix_prediction_data(zip_file_info_for_preprocessing)
    # generate_r2_chart(zip_file_info_for_preprocessing)
    # print(zip_file_info_for_preprocessing)

    with Pool(6) as p:
        p.map(partial(quantify_weight_variability, num_iter=100), zip_file_info_for_preprocessing)
        # p.map(partial(test_generalizability, num_iter=100, fluxnet_site_type='WSA'), zip_file_info_for_preprocessing)
    # for zf in zip_file_info_for_preprocessing:
        # quantify_variability(zf, 100, SimpleLSTMRegressor)
        # if zf[1] == 'US-Ton':
        #     generate_variability_graph(zf) 





