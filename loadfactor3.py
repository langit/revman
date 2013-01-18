'''
Try to measure in multiple ways.
'''

import noshow as nosh
from random import uniform
from copy import copy
import pylab

nosh.config.LBH=True
nosh.config.BINO=True
nosh.config.svl =0.001

nosh.config.samples = 1600
nosh.config.elites  = 140 
nosh.config.smoother = 0.7

#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('EMSR/HCR', 'EMSR/HAR', 'EMSR/NV', 'DP/LBH', 
		   'EMSR/SL', 'HCR/OSA', 'HAR/OSA')

def scenas(scen, ps):
	for p in ps:
		newscen = copy(scen)
		newscen.L = tuple(i*p for i in scen.L)
		newscen.U = tuple(i*p for i in scen.U)
		newscen.nid = newscen.demandFactor()
		print "## Scenario with LOAD Factor:", newscen.nid
		yield newscen

from ORinstance3 import sina

pickle = 'loadfactor3.pkl'
gld = [s for s in scenas(sina, (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1))]
#nosh.enuSim(gld, 10000, pickle, mymeth)
DISP = 'ORloadfactor3'
xlab = 'load factor'
nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
#nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
