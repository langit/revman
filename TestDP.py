'''
Test effect of: demand volatility
'''

import noshow as nosh
from random import uniform
from copy import copy
import pylab

nosh.config.LBH=True
nosh.config.BINO=True
nosh.config.svl =0.001

nosh.config.samples = 3000 #2900
nosh.config.elites  = 300 #200 
nosh.config.smoother = 0.7

#nosh.config.percentile = 0

#nosh.config.basemult = 4

#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('DP/LBH', 'EMSR/NV', 'EMSR/SL', 'HCR/OSA', 'HAR/OSA')

import sys
def scenas(scen, ps):
	mus = tuple((u+l)/2. for u,l in zip(scen.L, scen.U))
	for p in ps:
		newscen = copy(scen)
		newscen.L = (u-(u-l)*p for u,l in zip(mus,scen.L))
		newscen.L = tuple(l if l>=0 else 0. for l in newscen.L)
		newscen.U = tuple(u+(h-u)*p for u,h in zip(mus,scen.U))
		if 'normal' == sys.argv[-1]: 
			newscen = nosh.NormalScena(newscen)
		newscen.nid = p #/3**.5
		print "## Scenario volatility:", newscen.nid
		#print "Upper bound:", newscen.U
		yield newscen

from ORinstance3 import sina
#this line converts into normal demand distr

pickle = 'cov3.pkl'
gld = [s for s in scenas(sina, (i/50. for i in range(1,21)))]
for s in gld:
	print s.nid, s.policyStaticDP()
	#print s.nid, s.policyEMSR(139)
