'''
Test effect of:
Coefficient of variation 
'''

import noshow as nosh
from random import uniform
from copy import copy
import pylab

nosh.config.LBH=True
nosh.config.BINO=True
nosh.config.svl =0.001

nosh.config.samples = 400 #2900
nosh.config.elites  = 30 #200 
nosh.config.smoother = 0.7
nosh.config.iters = 31

#nosh.config.percentile = 0

#nosh.config.basemult = 4

#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('EMSR/HCR', 'EMSR/HAR', 'EMSR/NV', 'DP/LBH', 
		   'EMSR/SL', 'HCR/OSA', 'HAR/OSA')

def scenas(scen, ps):
	mus = tuple((u+l)/2. for u,l in zip(scen.L, scen.U))
	for p in ps:
		assert 0. <= p <= 1., "p out of range!"
		newscen = copy(scen)
		newscen.L = tuple(u-u*p for u in mus)
		newscen.U = tuple(u+u*p for u in mus)
		newscen.nid = p #/3**.5
		print "## Scenario with CV:", newscen.nid
		#print "Upper bound:", newscen.U
		yield newscen

#from ORinstance3 import sina
from ORinstance4 import sina

pickle = 'cov4easy.pkl'
gld = [s for s in scenas(sina, (0.5,.3,0.25,.2,0.15,.1,0.05,.0))]
nosh.enuSim(gld, 1000, pickle, mymeth)
DISP = 'cov4easy'
xlab = 'coefficient of variation'
nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle)) 
nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
