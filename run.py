from multiprocessing import Pool
from lstm_fluxnet_local import train_lstm_on_site
from ann_fluxnet_local import train_ann_on_site
from processing_utils import get_zip_info

if __name__ == '__main__':
	zip_file_info_for_preprocessing = get_zip_info("WSA")
	print(zip_file_info_for_preprocessing)
	# with Pool(5) as p:
	# 	p.map(train_lstm_on_site, zip_file_info_for_preprocessing)
	for zf in zip_file_info_for_preprocessing:
		if zf[1] == 'AU-Gin':
			train_lstm_on_site(zf)
			# train_ann_on_site(zf)


# idea for global scale: use first 75% of each sites data as training folds, then use the aggregate of 25% of each site as the test fold
