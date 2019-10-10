from multiprocessing import Pool
from lstm_fluxnet_local import train_lstm_on_site
from processing_utils import get_zip_info

if __name__ == '__main__':
	zip_file_info_for_preprocessing = get_zip_info()
	print(zip_file_info_for_preprocessing)
	with Pool(5) as p:
		p.map(train_lstm_on_site, zip_file_info_for_preprocessing)
