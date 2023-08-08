
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

s1= pd.Series({'1925-1934':1, '1935-1944':0, '1945-1954':0, '1955-1964':0,'1965-1974':0,'1975-1984':0, '1985-1994':0, '1995-2004':0,'2005-2014':2 })
s2= pd.Series({'1925-1934':0, '1935-1944':0, '1945-1954':0, '1955-1964':0,'1965-1974':1,'1975-1984':0, '1985-1994':0, '1995-2004':0,'2005-2014':0 })
s3= pd.Series({'1925-1934':0, '1935-1944':0, '1945-1954':0, '1955-1964':0,'1965-1974':0,'1975-1984':1, '1985-1994':0, '1995-2004':1,'2005-2014':2 })
s4= pd.Series({'1925-1934':0, '1935-1944':0, '1945-1954':0, '1955-1964':0,'1965-1974':0,'1975-1984':0, '1985-1994':1, '1995-2004':0,'2005-2014':0 })
s5= pd.Series({'1925-1934':1, '1935-1944':0, '1945-1954':0, '1955-1964':0,'1965-1974':0,'1975-1984':2, '1985-1994':0, '1995-2004':0,'2005-2014':0 })
s6= pd.Series({'1925-1934':0, '1935-1944':1, '1945-1954':0, '1955-1964':0,'1965-1974':0,'1975-1984':0, '1985-1994':0, '1995-2004':0,'2005-2014':0 })
s7= pd.Series({'1925-1934':0, '1935-1944':0, '1945-1954':0, '1955-1964':1,'1965-1974':0,'1975-1984':0, '1985-1994':2, '1995-2004':7,'2005-2014':8 })
s8= pd.Series({'1925-1934':0, '1935-1944':0, '1945-1954':0, '1955-1964':0,'1965-1974':0,'1975-1984':1, '1985-1994':0, '1995-2004':1,'2005-2014':0 })
s9= pd.Series({'1925-1934':1, '1935-1944':0, '1945-1954':0, '1955-1964':2,'1965-1974':1,'1975-1984':5, '1985-1994':11, '1995-2004':3,'2005-2014':8 })
s10= pd.Series({'1925-1934':0, '1935-1944':0, '1945-1954':0, '1955-1964':0,'1965-1974':0,'1975-1984':0, '1985-1994':0, '1995-2004':1,'2005-2014':0 })

df=pd.concat({'Hurricane/Typhoon/Tropical Storm':s1, 'Storms and Severe Thunderstorms':s2, 'Winter storm':s3, 'Storm surge':s4, 'Flood':s9, 'avalanche':s10,'Cold event':s5, 'Drought':s8,'Wild fire':s7,'Heat event':s6}, axis=1)

my_colors = ['b','Chocolate','cyan','Brown', 'LightPink','MediumVioletRed','DarkViolet','red', 'Indigo','SpringGreen']


fig=df.plot(kind='barh', stacked=True, color=my_colors, figsize=(14,8), xlim=(0,22), fontsize=20);
plt.savefig('/Users/emidia1/Desktop/newProjet/results/noDisastersNCan.png')

t1= pd.Series({'Flood': 126, 'Storms and Severe Thunderstorms': 47, 'Wildfire':41 ,'Winter Storm': 31, 'Tornado': 18,  'Drought': 14,'Avalanche': 8,  'Hurricane / Typhoon / Tropical Storm':8,'Cold Event': 7, 'Storm Surge':5 ,'Heat Event': 1})
t1.sort(axis=0)


fig = plt.figure(figsize=(14,8),dpi=300)
ax = fig.add_subplot(111)
ax.grid(True,which='both')
bar = ax.barh(range(1,t1.size+1,1),t1.values,0.6,align='center')
plt.yticks(range(1,t1.size+1,1), t1.index, size='small')
fig.tight_layout()


plt.savefig('/Users/emidia1/Desktop/newProjet/results/noDisastersCan.png')
