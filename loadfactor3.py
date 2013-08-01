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

nosh.config.samples = 3000
nosh.config.elites  = 300 
nosh.config.smoother = 0.7

#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('DP/LBH', 'EMSR/NV', 'EMSR/SL', 'HCR/OSA', 'HAR/OSA')

def scenas(scen, ps):
	for C in ps:
		newscen = copy(scen)
		newscen.C = newscen.nid = C 
		print "## LOAD Factor:", newscen.demandFactor()
		yield newscen

from ORinstance3 import sina

pickle = 'loadfactor3.pkl'
gld = [s for s in scenas(sina, range(100, 201, 10))]
nosh.enuSim(gld, 10000, pickle, mymeth)
DISP = 'ORloadfactor3'
xlab = 'total number of seats'
nosh.drawSubFigs(DISP, xlab, *nosh.loadResults(pickle))
nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
#nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
