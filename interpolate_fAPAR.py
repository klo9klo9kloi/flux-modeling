from scipy.io import loadmat
from scipy.interpolate import interp1d
import numpy as np 
import pandas as pd
from processing_utils import get_mat_info
import matplotlib.pyplot as plt

mat_info = get_mat_info()

for mat in mat_info:
	x = loadmat('modisfAPAR/' + mat)['FparData']
	df = pd.DataFrame(data=x, columns=['year', 'DOY', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
	# df['avg'] = np.mean(df.iloc[:, 2:], axis=1)
	# df = df[['year', 'DOY', 'avg']]
	x = list(range(0, len(df.index)*8, 8))
	y = np.mean(df.iloc[:, 2:], axis=1)
	f = interp1d(x, y)

	xnew = list(range(0, len(df.index)*8-7))
	# plt.plot(x, y, 'o', xnew, f(xnew))
	# plt.show()
	ynew = f(xnew)

	print(y[:2])
	print(ynew[:9])
	print(y[-2:])
	print(ynew[-9:])

	year = df['year'].astype('str')
	month = (df['DOY']/12).astype('int').astype('str')
	day = (df['DOY']%12).astype('str')

	for i in range(len(df.index)):
		if len(month[i]) == 1:
			month[i] = '0' + month[i]

	for i in range(len(df.index)):
		if len(day[i]) == 1:
			day[i] = '0' + day[i]

	new_frame = pd.DataFrame({'timestamp': year + month + day, 'avg': y})
	print(new_frame.head())
	break

