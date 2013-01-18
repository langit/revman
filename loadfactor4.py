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

nosh.config.buckets = False
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('EMSR/HCR', 'EMSR/HAR', 'EMSR/NV', 'DP/LBH', 
		   'EMSR/SL', 'EMSR/NO', 'HCR/OSA', 'HAR/OSA')

def scenas(scen, ps):
	for p in ps:
		newscen = copy(scen)
		newscen.L = tuple(i*p for i in scen.L)
		newscen.U = tuple(i*p for i in scen.U)
		newscen.nid = newscen.demandFactor()
		print "## Scenario with LOAD Factor:", newscen.nid
		yield newscen

sina = nosh.SimScena()
sina.m = 4
sina.L = (7., 15., 33., 10.)
sina.U = (37.,65., 103.,40.)
sina.f = (1050., 767., 527., 350.)
sina.C = 124.0
sina.beta = 0.1
sina.p = (0.1, 0.1)
sina.V = 1500.0
#vin = 5.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
#sina.L = (50.,85.)
#sina.U = (70.,115.)
print "Demand Factor: ", sina.demandFactor(), "V=", sina.V

pickle = 'loadfactor4.pkl'
gld = [s for s in scenas(sina, (.7, .8, .9, 1., 1.1, 1.2, 1.3))]
nosh.enuSim(gld, 20000, pickle, mymeth)
DISP = None#'inac-dist'
xlab = 'expected load factor'
nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
