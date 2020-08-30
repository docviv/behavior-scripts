import numpy as np
import re
import glob
import math
import csv
import matplotlib.pyplot as plt
from collections import Counter
from itertools import groupby
from scipy.stats import mstats
import pandas as pd

ffile = open('stats.txt', 'w')
newfile = open('pvalues.txt', 'w')
kruskallwallis = open('newpvalueshet1vshet2.txt', 'w')
for filename in sorted(glob.glob('linearmodel_*')):
	f = open(filename, 'r')
	lines = f.readlines()
	metric = []
	pvalues = []
	count = 0
	newfile.write(filename+"\n")
	for line in lines:
		if line.split(' ')[0] == 'anova:':
			if float(re.split(' |\n',line)[15]) < 0.05:
				count += 1
				metric.append(count)
				pvalues.append(float(re.split(' |\n',line)[15]))
				newfile.write(line.split(' ')[2] + ' '+ re.split(' |\n',line)[15]+'\n')

	for line in lines:
		if line.startswith("anova:"):
			continue
		else:
			if line.startswith("mutornot[T.wt] "):
				if len(line.split()) > 3:
					pvalue = line.split()[4]
					if float(pvalue) < 0.05:
						count += 1
						metric.append(count)
						pvalues.append(pvalue)

	namelist = []
	ssmdlist = []
#	for file in sorted(glob.glob('outputfulldata' + '_' + re.split('_',filename)[1]+'_'+re.split('_',filename)[2]+'_'+re.split('_',filename)[3]+'_'+re.split('_',filename)[4]+'_'+re.split('_',filename)[5]+'/*')):
	for file in sorted(glob.glob('outputfulldata_akt3_het1_vs_akt3_het2/*')):
		list = re.split("_",file)
		if "hom/boxgraph" in list:
			if "data" in list:
				df = pd.read_csv(file, sep='\t')
				arraywt1 = df[df.columns[1]].values
				arraymut = df[df.columns[2]].values
				try:
					H, pval = mstats.kruskalwallis(arraywt1, arraymut)
				except:
					continue
				kruskallwallis.write(file+' '+str(pval)+'\n')
		if "a2" in list:
			namelist.append(re.split("a2_data",file)[0])
	count1 = 0
	for name in namelist:
		if re.split('_',name)[7] == 'histgraph':
			continue
		arraywt = np.loadtxt(name + 'a1_data', delimiter = ',')
		arrayhom = np.loadtxt(name + 'a2_data', delimiter = ',')
		meanwt = np.nanmean(arraywt)
		meanmut = np.nanmean(arrayhom)
		varwt = np.nanvar(arraywt)
		varmut = np.nanvar(arrayhom)
		ssmd = (meanwt - meanmut) / (math.sqrt(varwt + varmut))
		if np.isnan(ssmd):
			continue
	#	if np.isinf(ssmd):
	#		continue
		if ssmd > 2:
			print(name+' '+str(ssmd))
		if ssmd < -2:
			print(name+' '+str(ssmd))
		ssmdlist.append(ssmd)
	count2 = 0
	count3 = 0
	metric = []
	count4 = 0
	for n,i in enumerate(ssmdlist):
		if i > 2:
			ssmdlist.pop(n)
			count2 += 1
	for n,i in enumerate(ssmdlist):
		if i < -2:
			ssmdlist.pop(n)
			count3 += 1
	for n,i in enumerate(ssmdlist):
		count4 += 1
		metric.append(count4)
	print(count4)
	ssmdarray = np.asarray(ssmdlist)
	ffile.write(filename+' '+str(np.mean(ssmdarray))+' '+str(np.median(ssmdarray))+' '+str(np.std(ssmdarray))+' '+str(len(ssmdarray))+' '+str(count)+' '+str(count2+count3)+'\n')
	with open(re.split('_',filename)[1]+'_'+re.split('_',filename)[2]+'_'+re.split('_',filename)[3]+'_'+re.split('_',filename)[4]+'_'+re.split('_',filename)[5]+'.npy', 'wb') as f:
		np.save(f, ssmdarray)
	
