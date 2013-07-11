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

nosh.config.samples = 2000
nosh.config.elites  = 200 
nosh.config.smoother = 0.5
nosh.config.iters = 51

#nosh.config.buckets = False
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('EMSR/NV', 'DP/LBH', 'EMSR/SL', 'HCR/OSA', 'HAR/OSA')

def scenas(scen, ps):
	for C in ps:
		newscen = copy(scen)
		#newscen.L = tuple(i*p for i in scen.L)
		#newscen.U = tuple(i*p for i in scen.U)
		newscen.nid = newscen.C = C
		print "## Scenario LOAD Factor:", newscen.demandFactor()
		yield newscen

from ORinstance4 import sina

pickle = 'loadfact4.pkl'
gld = [s for s in scenas(sina, range(80,161,10))]
nosh.enuSim(gld, 10000, pickle, mymeth)
DISP = 'loadfact4'
xlab = 'total number of seats'
nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
