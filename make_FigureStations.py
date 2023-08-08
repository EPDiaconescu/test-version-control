
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib  as mpl


coords_lon=[-124.7,-128.7,-134.9,-109.2,-113.7,-121.2,-112.0,-115.8,-133.5,-119.3,
            -129.0,-126.8,-125.3,-133.0,-125.6,-117.8,-123.4,-114.4,-123.7,-62.3,
            -96.1,-109.1,-105.1,-76.5,-61.6,-66.8,-90.7,-120.8,-68.5,-83.4,-71.2,
            -85.9,-63.8,-97.8,-81.2,-68.5,-101.7,-89.8,-115.1,-113.2,-75.1,-111.2,
            -85.7,-84.6,-95.0,-93.4,-78.1,-68.4,-77.8,-69.6,-139.1,-136.3,-139.1,
            -134.4,-137.5,-140.2,-135.9,-139.8,-137.4,-132.5,-137.2,-131.2,-132.7,
            -129.2,-128.8,-135.1,-57.0,-64.0,-60.4,-59.2,-55.8,-61.7,-66.9]

coords_lat=[70.2,66.2,67.4,62.7,61.2,61.8,60.0,60.8,68.3,76.2,69.9,65.3,72.0,69.5,
            64.9,70.8,63.2,62.5,61.2,82.5,64.3,68.8,69.1,64.2,66.6,68.5,63.3,69.6,70.5,64.2,
            68.7,80.0,67.5,68.7,68.8,63.7,68.7,68.5,67.8,68.5,68.9,65.8,68.3,73.0,74.7,68.8,
            58.5,58.1,55.3,61.1,61.4,62.1,64.0,62.2,60.7,69.6,63.6,67.6,62.8,62.0,69.0,60.0,
            60.2,60.9,60.1,60.7,53.7,53.5,53.3,55.1,52.3,56.6,52.9]



clevs = [-1500, 0, 200, 400, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2800, 3200, 3600, 4000]
cgbvr2 = mpl.colors.ListedColormap([[1., 1., .8], [.8, 1., .2], [.6, 1., 0.],  [0., 1., 0.],  [.2, .8, 0.], [0., .6, 0.], [.6, 1., 1.], [0., 1., 1.], [0., .6, 1.],  [0., 0., 1.], [0., 0., .6], [0., 0., .4], [.8, .6, 1.], [.8, .4, 1.], [.6, 0., 1.],  [.4, 0., .6], [1., 0., 0.], [.8, 0., 0.], [.6, 0., 0.], [.4, 0., 0.]])
cgbvr = mpl.colors.ListedColormap([[0., 1., 1.], [.8, 1., .2], [.6, 1., 0.],  [0., 1., 0.],  [.2, .8, 0.], [0., .6, 0.], [.6, 1., 1.], [0., 1., 1.], [0., .6, 1.],  [0., 0., 1.], [0., 0., .6], [0., 0., .4], [.8, .6, 1.], [.8, .4, 1.], [.6, 0., 1.],  [.4, 0., .6], [1., 0., 0.], [.8, 0., 0.], [.6, 0., 0.], [.4, 0., 0.]])



cmaps = [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]

points_tas=np.array([coords_lon,coords_lat])

fig1 = plt.figure(figsize=(11,8.5))
plt.subplots_adjust(left=0.02,right=0.98,top=0.90,bottom=0.10,wspace=0.05,hspace=0.05)
width = 7500000; height=4000000; lon_0 = -92; lat_0 = 70
m = Basemap(resolution='l',width=width,height=height,projection='aeqd', lat_0=lat_0,lon_0=lon_0)

m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawmeridians(np.arange(-180,180,20), linewidth=0.2)
m.drawparallels(np.arange(40,90,10),linewidth=0.2)

#################################

xt, yt = m(points_tas[0,:], points_tas[1,:])

m.shadedrelief()

m.plot(xt, yt, 'bo', markersize=10, lw = 0, markeredgecolor='k' )


plt.tight_layout()
plt.savefig('C:/Users/DiaconescuE/Desktop/Figure2_B.png')

plt.show()