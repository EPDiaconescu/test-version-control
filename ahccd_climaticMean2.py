#

import numpy as np
import numpy
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import cPickle as pickle
import matplotlib  as mpl
from matplotlib.colors import Normalize
from scipy import *

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


def save_pandas(fname, data):
    '''Save DataFrame or Series
    Parameters
    ----------
    fname : str
        filename to use
    data: Pandas DataFrame or Series
    '''
    np.save(open(fname, 'w'), data)
    if len(data.shape) == 2:
        meta = data.index,data.columns
    elif len(data.shape) == 1:
        meta = (data.index,)
    else:
        raise ValueError('save_pandas: Cannot save this type')
    s = pickle.dumps(meta)
    s = s.encode('string_escape')
    with open(fname, 'a') as f:
        f.seek(0, 2)
        f.write(s)

def load_pandas(fname, mmap_mode='r'):
    '''Load DataFrame or Series
    Parameters
    ----------
    fname : str
        filename
    mmap_mode : str, optional
        Same as numpy.load option
    '''
    values = np.load(fname, mmap_mode=mmap_mode)
    with open(fname) as f:
        numpy.lib.format.read_magic(f)
        numpy.lib.format.read_array_header_1_0(f)
        f.seek(values.dtype.alignment*values.size, 1)
        meta = pickle.loads(f.readline().decode('string_escape'))
    if len(meta) == 2:
        return pd.DataFrame(values, index=meta[0], columns=meta[1])
    elif len(meta) == 1:
        return pd.Series(values, index=meta[0])


nc_st = '/Volumes/Emilia/NowWorking/newProjet/data/ahccd/ahccd_stations/ahccd_anualMean_tas.nc'  # Your filename
nc_fid = Dataset(nc_st, 'r')  # Dataset is the class behavior to open the file
                             # and create an instance of the ncCDF4 class
nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)

field = nc_fid.variables['tas'][:]
time = nc_fid.variables['time'][:]
time_vectors = nc_fid.variables['time_vectors'][:]
lats0 = nc_fid.variables['lat'][:]
lons0 = nc_fid.variables['lon'][:]

df_f = pd.DataFrame(field)
df_t = pd.DataFrame(time_vectors)
df_t.columns = ['an', 'luna', 'ziua']

a = df_t.iloc[0,0]
b = df_t.iloc[-1,0]
c=range(a,b+1)
ani=[str(w) for w in c]
#    map(str,range(a,b+1))
df_f.index=ani

selection=df_f.loc['1979':'2005']


aaaa=selection.describe()
nrYY0=aaaa.loc[['count'],:]
media0=aaaa.loc[['mean'],:]
media=np.array(media0)
nrYY=np.array(nrYY0)
bar_max=max(media.T-272.15)
print('bar_max=',bar_max)
bar_min=min(media.T-272.15)
print('bar_min=',bar_min)

media[nrYY<20]=-999


clima=pd.DataFrame(media)
clima.index=['climaticMean_1979to2005']
lonsDF0=pd.DataFrame(lons0)
lonsDF=lonsDF0.T
lonsDF.index=['lons']

latsDF0=pd.DataFrame(lats0)
latsDF=latsDF0.T
latsDF.index=['lats']

new=clima.append(lonsDF)
final=new.append(latsDF)

save_pandas('/Volumes/Emilia/NowWorking/newProjet/data/ahccd/ahccd_stations/ahccd_tas_climaticMean-1979to2005', final)


lon_0=-97.0
o_lon_p=180.0
o_lat_p=42.5

fig = plt.figure(figsize=(10.6,8))
plt.subplots_adjust(left=0.02,right=0.98,top=0.90,bottom=0.10,wspace=0.05,hspace=0.05)
width = 8000000; height=5000000; lon_0 = -110; lat_0 = 65
map = Basemap(resolution='l',width=width,height=height,projection='aeqd',
            lat_0=lat_0,lon_0=lon_0)

map.drawcoastlines()
map.drawcountries(linewidth=1)
map.fillcontinents(color = [ 0.9        ,  0.9        ,  0.9])
map.drawmeridians(np.arange(-180,180,20),labels=[0,0,0,1], linewidth=0.2, fontsize=16)
map.drawparallels(np.arange(40,90,10),labels=[1,0,0,0], linewidth=0.2, fontsize=16)


cgbvr = mpl.colors.ListedColormap ([[0., 0., 0.],'Navy', 'Blue','DodgerBlue','LightSkyBlue',  'Lavender', 'LIGHTPink','LightCoral','Crimson','FireBrick','darkred' ])

#################################
data = []
data.append((lons0, lats0, media.T-272.15))

norm = Normalize()
cmap = cgbvr
lons = []
lats = []
mags = []
for lon, lat, mag in data:
    xpt,ypt = map(lon,lat)
    lons.append(xpt)
    lats.append(ypt)
    mags.append(mag)
x = np.array(lons)
y = np.array(lats)
#Create normalized magnitudes (range [0, 1.0]) for color map.
mag_norms = norm(np.array(mags))
#Create marker sizes as a function of magnitude.
z = (mag_norms * 0.0)+60.0
#z = (mag_norms * 10.0)**2.0
#Plot the data and create a colorbar.


increment=4
vmin=-24
vmax=vmin+increment*11



sc = map.scatter(x,y, s=z, zorder=4, marker='o', lw=.15, antialiased=True, cmap=cmap, c=mags, vmin=vmin, vmax=vmax)
#sc = map.scatter(x,y, s=z, zorder=4, marker='o', lw=.15, antialiased=True, cmap=cmap, c=mags, vmin=-1, vmax=14)

c = plt.colorbar(sc, orientation='horizontal', shrink=0.7, aspect=24, extend="max")
c.set_label("tas (C); 1979-2005 mean", size=20)
c.set_ticks(np.linspace(vmin+increment, vmax, 11))
new=[int(w) for w in np.linspace(vmin+increment, vmax, 11)]
#c.set_ticklabels(new)
font_size = 16 # Adjust as appropriate.
c.ax.tick_params(labelsize=font_size)

plt.savefig('/Volumes/Emilia/NowWorking/newProjet/data/ahccd/ahccd_stations/ahccd_tas_climaticMean-1979to2005.png')

plt.show()
