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

nosh.config.samples = 500
nosh.config.elites  = 40
nosh.config.smoother = 0.6

#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('EMSR/HCR', 'EMSR/HAR', 'EMSR/NV', 'DP/LBH', 
		   'EMSR/SL', 'EMSR/NO', 'HCR/OSA', 'HAR/OSA')

def scenas(scen, ps):
	for p in ps:
		newscen = copy(scen)
		newscen.p = (p,p)
		newscen.nid = p
		print "## Scenario with NSR:", p
		yield newscen

sina = nosh.SimScena()
sina.L = (20., 60.)
sina.U = (70.,150.)
sina.C = 124.0
sina.m = 2
sina.beta = 0.1
sina.p = (0.1, 0.1)
sina.f = (1050., 527.)
sina.V = 2500.0
#vin = 5.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
#sina.L = (50.,85.)
#sina.U = (70.,115.)
print "Demand Factor: ", sina.demandFactor(), "V=", sina.V

pickle = 'metrics2.pkl'
gld = [s for s in scenas(sina, (0.01, 0.05, 0.1, 0.15, 0.19))]
#nosh.glideSim(sina, sinb, 0, 11, 10.0, 10000, pickle, mymeth)
nosh.enuSim(gld, 10000, pickle, mymeth)
DISP = None#'inac-dist'
xlab = 'no-show probability'
nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
#nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
