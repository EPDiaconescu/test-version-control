import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MaxNLocator
from scipy import stats

#################################################
output= 'C:/Users/DiaconescuE/Desktop/'
fileName='test_figure2.png'
##############################################################

# Generate fake data 
# x can be values of T at the station
# y can be values of Td at the station
# q can be values of T in ERA5-Land
# w can be values of Td in ERA5-Land
# you can import these values with xarray and select the variable you need

x = np.random.normal(size=1000)
y = x * 3 + np.random.normal(size=1000)

q = np.random.normal(size=100)
w = q * 2 + np.random.normal(size=100)+2

datasetQW='ERA5-Land'
datasetXY='Station'
#########################################################

tt=np.concatenate((x, q), axis=0)
rr=np.concatenate((y, w), axis=0)

#get cor coef, linar regretion and line pasing by 0 for x, y
slope, intercept, r_value, p_value, std_err = stats.linregress(tt,rr)
print("r:", r_value)
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
plt.figure(1, figsize=(9.5,9))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)


# the scatter plot:
axScatter.scatter(x, y, s=18, color="red", marker="^", label=datasetXY)
axScatter.scatter(q, w, s=14, color="blue", marker="o", label=datasetQW)
axScatter.legend(loc=4)
axScatter.text(min(x)+.05*min(x),.95*max(y),'$R = %0.2f$'% r_value, fontsize=20)
axScatter.text(min(x)+.05*min(x),.75*max(y),'$Slope = %0.2f$'% cc, fontsize=20)
axScatter.plot(tl, rl, '-k',linewidth=2.0)
axScatter.plot(tl, rll, '--k',linewidth=2.0)
axScatter.axvline(linewidth=0.5, ls='--', color='gray')
axScatter.axhline(linewidth=0.5, ls='--', color='gray')
#Make the tickmarks pretty
ticklabels = axScatter.get_xticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('arial')

ticklabels = axScatter.get_yticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('arial')



# now determine nice limits by hand:
binwidth = 0.25

xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
limxy = (int(xymax/binwidth) + 1) * binwidth
qwmax = np.max([np.max(np.fabs(q)), np.max(np.fabs(w))])
limqw = (int(qwmax/binwidth) + 1) * binwidth


from scipy import stats
xxx = np.linspace(min(x), max(x), 1000)
yyy = np.linspace(min(y), max(y), 1000)
bins = np.arange(-limxy, limxy + binwidth, binwidth)
#axHistx.hist(x, normed=True,bins=bins,color = 'red',alpha = 0.05)
kde = stats.gaussian_kde(x)
axHistx.plot(xxx, kde(xxx),color = 'red',linewidth=2.0)
#axHisty.hist(y, normed=True,bins=bins, color = 'red',alpha = 0.05,orientation='horizontal')
kde = stats.gaussian_kde(y)
axHisty.plot(kde(yyy),yyy,color = 'red',linewidth=2.0)

qqq = np.linspace(min(q), max(q), 1000)
www = np.linspace(min(w), max(w), 1000)
bins = np.arange(-limqw, limqw + binwidth, binwidth)
#axHistx.hist(q, normed=True,bins=bins,color = 'blue',alpha = 0.05)
kde = stats.gaussian_kde(q)
axHistx.plot(qqq, kde(qqq),color = 'blue',linewidth=2.0)
#axHisty.hist(w, normed=True,bins=bins, color = 'blue',alpha = 0.05,orientation='horizontal')
kde = stats.gaussian_kde(w)
axHisty.plot(kde(yyy),yyy,color = 'blue',linewidth=2.0)


#Cool trick that changes the number of tickmarks for the histogram axes
axHisty.xaxis.set_major_locator(MaxNLocator(3))
axHistx.yaxis.set_major_locator(MaxNLocator(3))

ticklabels = axHistx.get_yticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('arial')

ticklabels = axHisty.get_xticklabels()
for label in ticklabels:
    label.set_fontsize(18)
    label.set_family('arial')

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

plt.savefig(output+fileName)
