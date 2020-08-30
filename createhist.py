import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
import math
import seaborn as sns

with open('npdatafiles/akt3box8_wt_vs_akt3box8_hom.npy', 'rb') as f:
	genoarray = np.load(f)
with open('npdatafiles/akt3_het1_vs_akt3_het2.npy', 'rb') as f:
	splitarray = np.load(f)
#with open('geno3.npy', 'rb') as f:
#	split2array = np.load(f)
#print("Geno Mode: "+str(mode(genoarray)))
#ax.hist(mydata, weights=np.zeros_like(mydata) + 1. / mydata.size)
w = 0.05
n1 = math.ceil((genoarray.max() - genoarray.min())/w)
n2 = math.ceil((splitarray.max() - splitarray.min())/w)
g = sns.distplot(genoarray, bins=n1, label='Hom vs WT')
x1,y1 = g.get_lines()[0].get_data()
#print(y1)
peakpos1 = x1[y1.argmax()]
print('Group peak: '+str(peakpos1))
#print(max(y1))
c = sns.distplot(splitarray, bins=n2, label='Het vs Het')
x2,y2 = c.get_lines()[1].get_data()
#print(y2)
peakpos2 = x2[y2.argmax()]
print('Control peak: '+str(peakpos2))
#print(max(y2))
#plt.hist(genoarray, bins=n1, weights=np.zeros_like(genoarray) + 1. / genoarray.size,  alpha=0.5, label='Geno Comparison')
#plt.hist(splitarray, bins=n2, weights=np.zeros_like(splitarray) + 1. / splitarray.size, alpha=0.5, label='Sibling comparison')
#plt.boxplot(genoarray)
#plt.boxplot(splitarray)
#plt.axvline(x=mode(genoarray.all()), color='k', linestyle='dashed', linewidth=1)
#plt.axvline(x=mode(splitarray.all()), color='k', linestyle='dashed', linewidth=1)
#print("Geno Mode: "+str(mode(genoarray)))
#print("Split Mode: "+str(mode(splitarray)))
#plt.hist(split2array, bins=100, alpha=0.6, label='Geno3')
plt.xlim(-1, 1)
plt.legend(loc='upper right')
#plt.title('Same Clutch')
plt.xlabel('SSMD values')
plt.ylabel('Probability Density')
plt.savefig("akt3histnew.png")
#ssmdarray = np.asarray(ssmdlist)
#hist, _ = np.histogram(ssmdarray[~np.isnan(ssmdarray)], range=(ssmdarray.min(),ssmdarray.max()))
#n, bins, patches = plt.hist(ssmdarray[~np.isnan(ssmdarray)], density =False, bins=100)
#print(len(bins))
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('My Very Own Histogram')
#maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.savefig("histo.png")
#print("Max ssmd value: "+str(max(ssmdlist)))
#print("Min ssmd value: "+str(min(ssmdlist)))
	#print(re.split("_",file))
	#if "a1" in list:
		#arraywt = np.loadtxt(file, delimiter = ',')
	#print(file)
	#if "a2" in list:
		#arrayhom = np.loadtxt(file, delimiter = ',')
		#print(file)
