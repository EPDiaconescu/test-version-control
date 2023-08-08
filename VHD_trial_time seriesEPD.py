# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:25:21 2019

@author: leeca
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
import scipy.stats as stats
import pickle as pickle
import os
import glob
import xarray
#from netCDF4 import Dataset
#import ncdump_python3 as ncdump
#%matplotlib inline
#mpl.rc('font',size=16) #set default font size and weight for plots

########RCP2.6###########
input = 'Y:/30. CLIMATE SERVICES DATA PRODUCTS OFFICE/05 - Personal/Lee/Decision Making Excercise/ASPHALT/data/VHD_counts/'
#output = 'H:/30. CLIMATE SERVICES DATA PRODUCTS OFFICE/05 - Personal/Lee/Decision Making Excercise/ASPHALT/data/VHD_outputs/'

os.chdir(input)

########RCP2.6###########
liste_26=glob.glob('*rcp26*.nc')

file_26=liste_26[0]
ds26 = xarray.open_dataset(file_26)
t26 = pd.to_datetime(ds26.time.values)
YYstring = pd.Series(t26.strftime('%Y'))
frame26=ds26['tasmax'].to_dataframe()
frame26.columns=['m1']
frame26.index = YYstring

for file_26, nr in zip(liste_26[1:], range(2, len(liste_26) + 2)):
	print (file_26, 'm' + str(nr))
	nameModel = 'm' + str(nr)
	ds26 = xarray.open_dataset(file_26)
	t26 = pd.to_datetime(ds26.time.values)
	YYstring = pd.Series(t26.strftime('%Y'))
	table26=ds26['tasmax'].to_dataframe()
	table26.columns=[nameModel]
	table26.index = YYstring
	frame26[nameModel]=table26


list_1_26=np.full([3], np.nan)
for year in frame26.index:
	QQ26=stats.mstats.mquantiles(frame26.loc[year],prob=[0.25, 0.5, 0.75],alphap=0.5,betap=0.5)
	list_1_26 = np.vstack([list_1_26, QQ26])
list_1_26=list_1_26[~np.isnan(list_1_26).any(axis=1)]
QQ26_table= pd.DataFrame(list_1_26)
QQ26_table.columns=['p25','p50','p75']
QQ26_table.index=frame26.index


########RCP4.5###########
liste_45=glob.glob('*rcp45*.nc')

file_45=liste_45[0]
ds45 = xarray.open_dataset(file_45)
t45 = pd.to_datetime(ds45.time.values)
YYstring = pd.Series(t45.strftime('%Y'))
frame45=ds45['tasmax'].to_dataframe()
frame45.columns=['m1']
frame45.index = YYstring

for file_45, nr in zip(liste_45[1:], range(2, len(liste_45) + 2)):
	print (file_45, 'm' + str(nr))
	nameModel = 'm' + str(nr)
	ds45 = xarray.open_dataset(file_45)
	t45 = pd.to_datetime(ds45.time.values)
	YYstring = pd.Series(t45.strftime('%Y'))
	table45=ds45['tasmax'].to_dataframe()
	table45.columns=[nameModel]
	table45.index = YYstring
	frame45[nameModel]=table45

list_1_45=np.full([3], np.nan)
for year in frame45.index:
	QQ45=stats.mstats.mquantiles(frame45.loc[year],prob=[0.25, 0.5, 0.75],alphap=0.5,betap=0.5)
	list_1_45 = np.vstack([list_1_45, QQ45])
list_1_45=list_1_45[~np.isnan(list_1_45).any(axis=1)]
QQ45_table= pd.DataFrame(list_1_45)
QQ45_table.columns=['p25','p50','p75']
QQ45_table.index=frame45.index


########RCP8.5###########
liste_85=glob.glob('*rcp85*.nc')

file_85=liste_85[0]
ds85 = xarray.open_dataset(file_85)
t85 = pd.to_datetime(ds85.time.values)
YYstring = pd.Series(t85.strftime('%Y'))
frame85=ds85['tasmax'].to_dataframe()
frame85.columns=['m1']
frame85.index = YYstring

for file_85, nr in zip(liste_85[1:], range(2, len(liste_85) + 2)):
	print (file_85, 'm' + str(nr))
	nameModel = 'm' + str(nr)
	ds85 = xarray.open_dataset(file_85)
	t85 = pd.to_datetime(ds85.time.values)
	YYstring = pd.Series(t85.strftime('%Y'))
	table85=ds85['tasmax'].to_dataframe()
	table85.columns=[nameModel]
	table85.index = YYstring
	frame85[nameModel]=table85

list_1_85=np.full([3], np.nan)
for year in frame85.index:
	QQ85=stats.mstats.mquantiles(frame85.loc[year],prob=[0.25, 0.5, 0.75],alphap=0.5,betap=0.5)
	list_1_85 = np.vstack([list_1_85, QQ85])
list_1_85=list_1_85[~np.isnan(list_1_85).any(axis=1)]
QQ85_table= pd.DataFrame(list_1_85)
QQ85_table.columns=['p25','p50','p75']
QQ85_table.index=frame85.index



##############################

rcp26_25=QQ26_table['p25'].values
rcp26_50=QQ26_table['p50'].values
rcp26_75=QQ26_table['p75'].values

rcp45_25=QQ45_table['p25'].values
rcp45_50=QQ45_table['p50'].values
rcp45_75=QQ45_table['p75'].values

rcp85_25=QQ85_table['p25'].values
rcp85_50=QQ85_table['p50'].values
rcp85_75=QQ85_table['p75'].values

years=(frame85.index).values

plt.figure(figsize=(10,6))

plt.plot(years, rcp26_50, color='blue', label='RCP 4.5 50th percentile')
plt.fill_between(years, rcp26_25, rcp26_75, color='blue', alpha=0.3, label='RCP4.5')

plt.plot(years, rcp85_50, color='red', label='RCP 4.5 50th percentile')
plt.fill_between(years, rcp85_25, rcp85_75, color='red', alpha=0.3, label='RCP4.5')

plt.plot(years, rcp45_50, color='orange', label='RCP 4.5 50th percentile')
plt.fill_between(years, rcp45_25, rcp45_75, color='orange', alpha=0.3, label='RCP4.5')


plt.xticks(['1960', '1980', '2000', '2020', '2040', '2060', '2080', '2100'])
plt.ylabel('Number of Very Hot Days')
plt.legend()
plt.ylim(0, 150)


##############2030s#######################
timep=['2021-2050', '2051-2080']
rcp=['rcp26','rcp45','rcp85']



plt.figure(figsize=(10,6))
#file_='Y:/30. CLIMATE SERVICES DATA PRODUCTS OFFICE/05 - Personal/Hernandez/decision making exercise/extreme hot days/rcp26/2021-2050/climateMean_2021-2050_tasmax.csv'
#df_ = pd.read_csv(file_,index_col=None, header=0)
#plt.boxplot(df_['value'])
plt.figure(figsize=(10,6))
plt.legend()

