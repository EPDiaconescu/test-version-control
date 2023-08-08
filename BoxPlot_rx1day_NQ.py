import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pylab
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes
import matplotlib.patches as mpatches

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

def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue',facecolor='blue')
    setp(bp['caps'][0], color='blue',linewidth=2)
    setp(bp['caps'][1], color='blue',linewidth=2)
    setp(bp['whiskers'][0], color='blue',linestyle='-', linewidth=2)
    setp(bp['whiskers'][1], color='blue',linestyle='-', linewidth=2)
    setp(bp['fliers'][0],  marker='*',markersize=8, markeredgecolor='blue', markerfacecolor='blue')
    setp(bp['medians'][0], color='white',linewidth=3)

    setp(bp['boxes'][1], color='red',facecolor='red')
    setp(bp['caps'][2], color='red',linewidth=2)
    setp(bp['caps'][3], color='red',linewidth=2)
    setp(bp['whiskers'][2], color='red',linestyle='-', linewidth=2)
    setp(bp['whiskers'][3], color='red',linestyle='-', linewidth=2)
    setp(bp['fliers'][1], marker='*',markersize=8, markeredgecolor='red', markerfacecolor='red')
    setp(bp['medians'][1], color='white',linewidth=3)



##############################################################

# Generate the array
#x delta pr H50 RCP45; y delta pr H50 RCP85
#z delta pr H85 RCP45; w delta pr H85 RCP85

domeniu='NQ'
varName='rx1day'
#oyLabel=r'$\Delta$'+"  T ($^\circ$C)"
oyLabel=r'$\Delta$'+" RX1day [mm/day] \n"

periode1='2040to2064'
periode2='2076to2100'
input_45= 'D:/NowWorking/newProjet_2017/data_models/rcp45_NQ_mk_fldMean/'
input_85= 'D:/NowWorking/newProjet_2017/data_models/rcp85_NQ_mk_fldMean/'
output= 'D:/NowWorking/newProjet_2017/results_ArticNET_IRIS4/boxplot_indices_NQ/'

listSim_45 = ['CCCma-CanRCM4_NA22_CanESM2','DMI-HIRHAM5_NA44_EC-EARTH','SMHI-RCA4_NA44_CanESM2', 'SMHI-RCA4_NA44_EC-EARTH',
              'UQAM-CRCM5_NA44_CanESM2','UQAM-CRCM5_NA44_MPI-ESM-LR','OURANOS-CRCM5_NA22_CanESM2m1']

listSim_85 = ['CCCma-CanRCM4_NA22_CanESM2','DMI-HIRHAM5_NA44_EC-EARTH', 'SMHI-RCA4_NA44_CanESM2','SMHI-RCA4_NA44_EC-EARTH',
              'UQAM-CRCM5_NA44_CanESM2', 'UQAM-CRCM5_NA44_MPI-ESM-LR','OURANOS-CRCM5_NA22_CanESM2ens',
              'IowaState-RegCM4_NA22_GFDL-ESM2M','IowaState-RegCM4_NA22_HadGEM2-ES','NCAR-RegCM4_NA22_MPI-M-MPI-ESM-LR',
              'UArizona-WRF2_NA22_GFDL-ESM2M']


y=[]
for file in listSim_85:
    ncMK=input_85+varName+'_'+file+'_rcp85_delta_'+periode1+'_NQ_mk_fldMean.nc'
    nc_fidMK = Dataset(ncMK, 'r')
    modelMK=nc_fidMK.variables[varName][:].squeeze()
    y=np.append(y, modelMK)
    nc_fidMK.close()
df_new=pd.DataFrame(y)
df_new.index=listSim_85
df_new.columns=['rcp85_H50']

w=[]
for file in listSim_85:
    ncMK=input_85+varName+'_'+file+'_rcp85_delta_'+periode2+'_NQ_mk_fldMean.nc'
    nc_fidMK = Dataset(ncMK, 'r')
    modelMK=nc_fidMK.variables[varName][:].squeeze()
    w=np.append(w, modelMK)
    nc_fidMK.close()
rcp85_H85=pd.DataFrame(w)
rcp85_H85.index=listSim_85
df_new['rcp85_H85']=rcp85_H85

x=[]
for file in listSim_45:
    ncMK=input_45+varName+'_'+file+'_rcp45_delta_'+periode1+'_NQ_mk_fldMean.nc'
    nc_fidMK = Dataset(ncMK, 'r')
    modelMK=nc_fidMK.variables[varName][:].squeeze()
    x=np.append(x, modelMK)
    nc_fidMK.close()
x1=np.append(x, np.nan)
x1=np.append(x1, np.nan)
x1=np.append(x1, np.nan)
x1=np.append(x1, np.nan)
rcp45_H50=pd.DataFrame(x1)
rcp45_H50.index=listSim_85
df_new['rcp45_H45']=rcp45_H50

z=[]
for file in listSim_45:
    ncMK=input_45+varName+'_'+file+'_rcp45_delta_'+periode2+'_NQ_mk_fldMean.nc'
    nc_fidMK = Dataset(ncMK, 'r')
    modelMK=nc_fidMK.variables[varName][:].squeeze()
    z=np.append(z, modelMK)
    nc_fidMK.close()
z1=np.append(z, np.nan)
z1=np.append(z1, np.nan)
z1=np.append(z1, np.nan)
z1=np.append(z1, np.nan)
rcp45_H85=pd.DataFrame(z1)
rcp45_H85.index=listSim_85
df_new['rcp45_H85']=rcp45_H85


q=[]
for file in listSim_85:
    ncMK=input_85+varName+'_'+file+'_rcp85_1980to2004Mean_NQ_mk_fldMean.nc'
    nc_fidMK = Dataset(ncMK, 'r')
    modelMK=nc_fidMK.variables[varName][:].squeeze()
    q=np.append(q, modelMK)
    nc_fidMK.close()
rcp85_R=pd.DataFrame(q)
rcp85_R.index=listSim_85
df_new['rcp85_Ref']=rcp85_R

df_newM=(pd.DataFrame(df_new.median(axis=0))).transpose()
df_newM.index=['Mediane']
df_newT=df_new.append(df_newM)
df_newT.to_csv('D:/NowWorking/newProjet_2017/results_ArticNET_IRIS4/txt_delta_NQ/'+varName+'_deltaNQmean_eachRCM.csv')
print df_newT


#########################################################



fig = figure(figsize=(8,6), dpi=400)
ax = axes()
hold(True)

dataH50 = [x,y]
dataH85 = [z,w]

boxX=boxplot(dataH50, positions = [1.3, 1.7], patch_artist=True,widths = 0.3)
setBoxColors(boxX)
boxX=boxplot(dataH85, positions = [3.3,3.7], patch_artist=True,widths = 0.3)
setBoxColors(boxX)

ax.set_xticks([1.5, 3.5])
ax.set_xticklabels(['H 50', 'H 85'],fontsize=16, family='arial')
ticklabels = ax.get_yticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('arial')

xlim(0,5)

# draw temporary red and blue lines and use them to create a legend
blue_patch = mpatches.Patch(color='blue', label='RCP 4.5')
red_patch = mpatches.Patch(color='red', label='RCP 8.5')

plt.legend(handles=[blue_patch, red_patch],fontsize=16,loc=2)
plt.ylabel(oyLabel, fontsize=20)
#plt.ylim(0, 24)
plt.savefig(output+varName+'_boxplot_'+domeniu+'_EG.png',bbox_inches='tight', pad_inches=0.3, dpi=400)
