import datetime as dt  # Python standard library datetime  module
import numpy as np
from netCDF4 import Dataset,  num2date
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib  as mpl
from matplotlib.colors import Normalize
norm = Normalize()
import pandas as pd

def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print "\t\ttype:", repr(nc_fid.variables[key].dtype)
            for ncattr in nc_fid.variables[key].ncattrs():
                print '\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr))
        except KeyError:
            print "\t\tWARNING: %s does not contain variable attributes" % key

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print "NetCDF Global Attributes:"
        for nc_attr in nc_attrs:
            print '\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print "NetCDF dimension information:"
        for dim in nc_dims:
            print "\tName:", dim
            print "\t\tsize:", len(nc_fid.dimensions[dim])
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print "NetCDF variable information:"
        for var in nc_vars:
            if var not in nc_dims:
                print '\tName:', var
                print "\t\tdimensions:", nc_fid.variables[var].dimensions
                print "\t\tsize:", nc_fid.variables[var].size
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def smooth(x,beta):
    """ kaiser window smoothing """
    window_len=11
    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w = np.kaiser(window_len,beta)
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[5:len(y)-5]

var='tasAnualMean'
labelYY=' Annual mean T [$^\circ$C]'
#labelUp=''



input= 'D:/NowWorking/newProjet_2017/data_models/indices_models/'+var+'/'
inputMK='D:/NowWorking/newProjet_2017/data_models/All_masks/'

output= 'D:/NowWorking/newProjet_2017/results_MFFP/timeSeries-plot/'

listSim1 = ['CCCma-CanRCM4_NA22_CanESM2','DMI-HIRHAM5_NA44_EC-EARTH', 'SMHI-RCA4_NA44_CanESM2','SMHI-RCA4_NA44_EC-EARTH',
              'UQAM-CRCM5_NA44_CanESM2', 'UQAM-CRCM5_NA44_MPI-ESM-LR']
listSim2 = ['CCCma-CanRCM4_NA22_CanESM2','DMI-HIRHAM5_NA44_EC-EARTH', 'SMHI-RCA4_NA44_CanESM2','SMHI-RCA4_NA44_EC-EARTH',
              'UQAM-CRCM5_NA44_CanESM2', 'UQAM-CRCM5_NA44_MPI-ESM-LR', 'OURANOS-CRCM5_NA22_CanESM2ens']



gmfd=['gmfd_reanalysis_short_allYY']
ncMK=inputMK+'mask_NQ_gmfd_reanalysis_short.nc'
print ncMK
nc_fidMK = Dataset(ncMK, 'r')
modelMK=nc_fidMK.variables['mask_qc'][:,:].squeeze()
nc_fidMK.close()

nc0=input+var+'_'+gmfd[0]+'.nc'
nc0_fid = Dataset(nc0, 'r')
modelVar0 = nc0_fid.variables[var[:-9]][:,:,:].squeeze()
lats = nc0_fid.variables['lat'][:]
lons = nc0_fid.variables['lon'][:]
time_m= nc0_fid.variables['time'][:]
units = nc0_fid.variables['time'].units
calendar = nc0_fid.variables['time'].calendar
dates = num2date(time_m[:],units=units,calendar=calendar)
new_index_sh0=[int(date.year) for date in dates]
nc0_fid.close()

modelVar=modelVar0+modelMK
data_shDF0=pd.DataFrame(modelVar.mean(axis=1).mean(axis=1), columns=['GMFD'], index=new_index_sh0)
ref=data_shDF0.copy()
reference=(data_shDF0.loc[1980:2004]).mean()

for file in listSim1:
    ncMK=inputMK+'mask_NQ_'+file+'.nc'
    nc_fidMK = Dataset(ncMK, 'r')
    masca=nc_fidMK.variables['mask_qc'][:,:].squeeze()
    nc_fidMK.close()

    nc1 = input+var+'_'+file+'_histo_allYY.nc'
    nc1_fid = Dataset(nc1, 'r')
    modelVar1 = nc1_fid.variables[var[:-9]][:,:,].squeeze()
    lat2d = nc1_fid.variables['lat'][:]
    lon2d = nc1_fid.variables['lon'][:]
    time_m= nc1_fid.variables['time'][:]
    units = nc1_fid.variables['time'].units
    calendar = nc1_fid.variables['time'].calendar
    dates = num2date(time_m[:],units=units,calendar=calendar)
    new_index_sh1=[int(date.year) for date in dates]
    nc1_fid.close()

    modelVar1=modelVar1+masca
    data_shDF1=pd.DataFrame(modelVar1.mean(axis=1).mean(axis=1), columns=[file+'_rcp45'], index=new_index_sh1)

    nc2 = input+var+'_'+file+'_rcp45_allYY.nc'
    nc2_fid = Dataset(nc2, 'r')
    modelVar2 = nc2_fid.variables[var[:-9]][:,:,].squeeze()
    time_m= nc2_fid.variables['time'][:]
    units = nc2_fid.variables['time'].units
    calendar = nc2_fid.variables['time'].calendar
    dates = num2date(time_m[:],units=units,calendar=calendar)
    new_index_sh2=[int(date.year) for date in dates]
    nc2_fid.close()

    modelVar2=modelVar2+masca
    data_shDF2=pd.DataFrame(modelVar2.mean(axis=1).mean(axis=1), columns=[file+'_rcp45'], index=new_index_sh2)

    modelMK=pd.concat([data_shDF1,data_shDF2], axis=0)
    data_shDF0=pd.concat([data_shDF0,modelMK],axis=1)

for file in listSim2:
    ncMK=inputMK+'mask_NQ_'+file+'.nc'
    nc_fidMK = Dataset(ncMK, 'r')
    masca=nc_fidMK.variables['mask_qc'][:,:].squeeze()
    nc_fidMK.close()

    nc1 = input+var+'_'+file+'_histo_allYY.nc'
    nc1_fid = Dataset(nc1, 'r')
    modelVar1 = nc1_fid.variables[var[:-9]][:,:,].squeeze()
    lat2d = nc1_fid.variables['lat'][:]
    lon2d = nc1_fid.variables['lon'][:]
    time_m= nc1_fid.variables['time'][:]
    units = nc1_fid.variables['time'].units
    calendar = nc1_fid.variables['time'].calendar
    dates = num2date(time_m[:],units=units,calendar=calendar)
    new_index_sh1=[int(date.year) for date in dates]
    nc1_fid.close()

    modelVar1=modelVar1+masca
    data_shDF1=pd.DataFrame(modelVar1.mean(axis=1).mean(axis=1), columns=[file+'_rcp85'], index=new_index_sh1)

    nc2 = input+var+'_'+file+'_rcp85_allYY.nc'
    nc2_fid = Dataset(nc2, 'r')
    modelVar2 = nc2_fid.variables[var[:-9]][:,:,].squeeze()
    time_m= nc2_fid.variables['time'][:]
    units = nc2_fid.variables['time'].units
    calendar = nc2_fid.variables['time'].calendar
    dates = num2date(time_m[:],units=units,calendar=calendar)
    new_index_sh2=[int(date.year) for date in dates]
    nc2_fid.close()

    modelVar2=modelVar2+masca
    data_shDF2=pd.DataFrame(modelVar2.mean(axis=1).mean(axis=1), columns=[file+'_rcp85'], index=new_index_sh2)

    modelMK=pd.concat([data_shDF1,data_shDF2], axis=0)
    data_shDF0=pd.concat([data_shDF0,modelMK],axis=1)


print('construct matrix')

anomalies=data_shDF0.sub((data_shDF0.loc[1980:2004]).mean(axis=0), axis=1)
corrected=(anomalies+reference[0])-273.15
ref=corrected['GMFD']

df_rcp45=corrected[['CCCma-CanRCM4_NA22_CanESM2_rcp45',
 'DMI-HIRHAM5_NA44_EC-EARTH_rcp45',
 'SMHI-RCA4_NA44_CanESM2_rcp45',
 'SMHI-RCA4_NA44_EC-EARTH_rcp45',
 'UQAM-CRCM5_NA44_CanESM2_rcp45',
 'UQAM-CRCM5_NA44_MPI-ESM-LR_rcp45']]
df_rcp85=corrected[['CCCma-CanRCM4_NA22_CanESM2_rcp85',
 'DMI-HIRHAM5_NA44_EC-EARTH_rcp85',
 'SMHI-RCA4_NA44_CanESM2_rcp85',
 'SMHI-RCA4_NA44_EC-EARTH_rcp85',
 'UQAM-CRCM5_NA44_CanESM2_rcp85',
 'UQAM-CRCM5_NA44_MPI-ESM-LR_rcp85',
 'OURANOS-CRCM5_NA22_CanESM2ens_rcp85']]

df_rcp45_med=df_rcp45.median(axis=1)
df_rcp45_min=df_rcp45.min(axis=1)
df_rcp45_max=df_rcp45.max(axis=1)

df_rcp85_med=df_rcp85.median(axis=1)
df_rcp85_min=df_rcp85.min(axis=1)
df_rcp85_max=df_rcp85.max(axis=1)

fig = plt.figure(figsize=(18,8))
ax = fig.add_subplot(111)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='both',length=10, width=1.0)

ax.fill_between((df_rcp85.loc[1950:2005]).index.tolist(), df_rcp85_min.loc[1950:2005], df_rcp85_max.loc[1950:2005], alpha=.2, color='k',label=' RCM historical ensemble')
ax.fill_between((df_rcp45.loc[2005:2100]).index.tolist(), df_rcp45_min.loc[2005:2100], df_rcp45_max.loc[2005:2100], alpha=.2, color='b',label=' RCM RCP 4.5 ensemble')
ax.fill_between((df_rcp85.loc[2005:2100]).index.tolist(), df_rcp85_min.loc[2005:2100], df_rcp85_max.loc[2005:2100], alpha=.2, color='r',label=' RCM RCP 8.5 ensemble')

plt.plot(df_rcp45.index.tolist(), smooth(df_rcp45_med,2), 'b-', linewidth=2.5)
plt.plot(df_rcp85.index.tolist(), smooth(df_rcp85_med,2), 'r-', linewidth=2.5)
plt.plot(ref.index.tolist(), ref, 'ko-', linewidth=1.5, label=' GMFD')
plt.axhline(linewidth=1.0, ls='--', color='k')
plt.legend(loc=2, fontsize=16,frameon=False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(1950, 2100)
plt.ylabel(labelYY, fontsize=16)
plt.xlabel('Years', fontsize=16)
#plt.title(labelUp, fontsize=16)
#plt.ylim(ylmi, ylma)

#fig.autofmt_xdate()
plt.savefig(output+'timeSerie_'+var+'.png', dpi=400)

plt.show()





