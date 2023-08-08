import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MaxNLocator
from scipy import stats
from netCDF4 import Dataset,  num2date
import pylab


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

def suplabel(axis,label,label_prop=None,
             labelpad=5,
             ha='center',va='center'):
    ''' Add super ylabel or xlabel to the figure
    Similar to matplotlib.suptitle
    axis       - string: "x" or "y"
    label      - string
    label_prop - keyword dictionary for Text
    labelpad   - padding from the axis (default: 5)
    ha         - horizontal alignment (default: "center")
    va         - vertical alignment (default: "center")
    '''
    fig = pylab.gcf()
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin,ymin = min(xmin),min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation=90.
        x = xmin-float(labelpad)/dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad)/dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None:
        label_prop = dict()
    pylab.text(x,y,label,rotation=rotation,
               transform=fig.transFigure,
               ha=ha,va=va,fontsize=20,
               **label_prop)
##############################################################

# Generate the array
#x delta temperature RCP85; y delta precip RCP85
#q delta temperature RCP45; w delta precip RCP45

domeniu='NQ'
rcp='rcp85'
periode='2040to2064'

input1= 'D:/NowWorking/newProjet_2017/data_models/CMIP5_domainMean_NQ_rcp85/'
input2= 'D:/NowWorking/newProjet_2017/data_models/domainMean_NQ_rcp85/'
output= 'D:/NowWorking/newProjet_2017/results_ArticNET_IRIS4/figures_bivariates_NQ/'

listSim1 = ['m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12', 'm13', 'm14', 'm15', 'm16', 'm17', 'm18', 'm19',
            'm20', 'm21', 'm22', 'm23', 'm24', 'm25', 'm26', 'm27', 'm28', 'm29', 'm30', 'm31', 'm32', 'm33', 'm34', 'm35', 'm36', 'm37', 'm38']

listSim2 = ['CCCma-CanRCM4_NA22_CanESM2', 'DMI-HIRHAM5_NA44_EC-EARTH', 'IowaState-RegCM4_NA22_GFDL-ESM2M', 'IowaState-RegCM4_NA22_HadGEM2-ES',
            'NCAR-RegCM4_NA22_MPI-M-MPI-ESM-LR', 'OURANOS-CRCM5_NA22_CanESM2ens','UArizona-WRF2_NA22_GFDL-ESM2M',
            'SMHI-RCA4_NA44_CanESM2', 'SMHI-RCA4_NA44_EC-EARTH', 'UQAM-CRCM5_NA44_CanESM2', 'UQAM-CRCM5_NA44_MPI-ESM-LR']

minT=0
maxT=8
minPR=0
maxPR=0.8

q=[]
for file in listSim1:
    ncMK=input1+'tasAnualMean_CMIP5_NA_'+file+'_'+rcp+'_fldMean_mk_delta_'+periode+'.nc'
    nc_fidMK = Dataset(ncMK, 'r')
    modelMK=nc_fidMK.variables['tas'][:].squeeze()
    q=np.append(q, modelMK)
    nc_fidMK.close()

w=[]
for file in listSim1:
    ncMK=input1+'prAnualMean_CMIP5_NA_'+file+'_'+rcp+'_fldMean_mk_delta_'+periode+'.nc'
    nc_fidMK = Dataset(ncMK, 'r')
    modelMK=nc_fidMK.variables['pr'][:].squeeze()
    w=np.append(w, modelMK*86400)
    nc_fidMK.close()

x=[]
for file in listSim2:
    ncMK=input2+'tasAnualMean_'+file+'_'+rcp+'_fldMean_mk_delta_'+periode+'.nc'
    nc_fidMK = Dataset(ncMK, 'r')
    modelMK=nc_fidMK.variables['tas'][:].squeeze()
    x=np.append(x, modelMK)
    nc_fidMK.close()

y=[]
for file in listSim2:
    ncMK=input2+'prAnualMean_'+file+'_'+rcp+'_fldMean_mk_delta_'+periode+'.nc'
    nc_fidMK = Dataset(ncMK, 'r')
    modelMK=nc_fidMK.variables['pr'][:].squeeze()
    y=np.append(y, modelMK)
    nc_fidMK.close()


#########################################################

tt=np.concatenate((x, q), axis=0)
rr=np.concatenate((y, w), axis=0)

#get cor coef, linar regretion and line pasing by 0 for x, y
slope, intercept, r_value, p_value, std_err = stats.linregress(tt,rr)
print "r:", r_value
tl = [min(tt), max(tt)]
rl = [slope*xx + intercept  for xx in tl]
cc=np.median(rr/tt)
rll=[cc*xx  for xx in tl]



nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.6
bottom, height = 0.1, 0.6
bottom_h = left_h = left + width + 0.04

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(11,11))


axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# #LINIILE PENTRU KDE 2D
# xmin, xmax = min(tt), max(tt)
# ymin, ymax = min(rr), max(rr)
# # Perform a kernel density estimate (KDE) on the data
# X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.vstack([X.ravel(), Y.ravel()])
# values = np.vstack([tt, rr])
# kernel = stats.gaussian_kde(values)
# f = np.reshape(kernel(positions).T, X.shape)
# axScatter.contour(X,Y,f,3,linestyles='dashed', colors='k',linewidths=2.5,)


# the scatter plot:

axScatter.scatter(q, w, s=62, color="blue", marker="o")
axScatter.scatter(x, y, s=80, color="red", marker="o")
#for i, txt in enumerate(stxt):
#    axScatter.annotate(txt, (x[i]+0.01,y[i]+0.005),color='r')




#axScatter.plot([minT, maxT], [minPR, maxPR], '--k',linewidth=2.0)
axScatter.set_xlabel(r"$\Delta$  Temperature ($^\circ$ C)", fontsize=20)
axScatter.set_ylabel(r"$\Delta$  Precipitation (mm/day)", fontsize=20)


#Make the tickmarks pretty
ticklabels = axScatter.get_xticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('arial')

ticklabels = axScatter.get_yticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('arial')




dataX = [q,x]
dataY = [w,y]
boxX=axHistx.boxplot(dataX,vert=0,patch_artist=True,widths = [0.25, 0.25])
boxY=axHisty.boxplot(dataY,patch_artist=True,widths = [0.25, 0.25])

#Cool trick that changes the number of tickmarks for the histogram axes
colors = ['blue', 'red']
for patch, color in zip(boxX['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_color(color)
for patch, color in zip(boxY['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_color(color)

for patch, color in zip(boxX['whiskers'], ['blue', 'blue','red','red']):
    patch.set_color(color)
for patch, color in zip(boxY['whiskers'], ['blue', 'blue','red','red']):
    patch.set_color(color)

for whisker in boxX['whiskers']:
    whisker.set(linestyle='-', linewidth=2)
for whisker in boxY['whiskers']:
    whisker.set(linestyle='-',linewidth=2)

for patch, color in zip(boxX['caps'], ['blue', 'blue','red','red']):
    patch.set_color(color)
for patch, color in zip(boxY['caps'], ['blue', 'blue','red','red']):
    patch.set_color(color)

for cap in boxX['caps']:
    cap.set( linewidth=2)
for cap in boxY['caps']:
    cap.set( linewidth=2)

for median in boxX['medians']:
    median.set(color='w', linewidth=4)
for median in boxY['medians']:
    median.set(color='w', linewidth=4)


for flier in boxX['fliers']:
    flier.set(marker='*', markeredgecolor='blue', markerfacecolor='blue')
for flier in boxY['fliers']:
    flier.set(marker='*', markeredgecolor='red', markerfacecolor='red')

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

axHistx2 = axHistx.twiny()
axHisty2 = axHisty.twinx()

axHistx2.set_xlim(axScatter.get_xlim())
axHisty2.set_ylim(axScatter.get_ylim())

axHisty.set_xticklabels(['CMIP5', 'CORDEX'], fontsize=16, rotation=90)
axHistx.set_yticklabels(['CMIP5', 'CORDEX'], fontsize=16)

ticklabels = axHistx2.get_xticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('arial')

ticklabels = axHisty2.get_yticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('arial')

plt.savefig(output+'PRvsTMEAN_'+domeniu+'_'+rcp+'_'+periode+'.png')
plt.show()