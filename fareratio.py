import noshow as nosh
from random import uniform
from copy import copy
import pylab

sina = nosh.SimScena()
sina.C = 90.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.1,0.1)
sina.f = (100.,40.)
sina.V = 300.0
sina.U = (75.,120.)
sina.L = (25.,60.)
print "Demand Factoer: ", sina.demandFactor(), sina.V


nosh.config.LBH=True
nosh.config.BINO=True
nosh.config.svl =0.001

nosh.config.samples = 2000 #2900
nosh.config.elites  = 200 #200 
nosh.config.smoother = 0.5
nosh.config.iters = 51

mymeth = ('DP/LBH', 'EMSR/NV', 'EMSR/SL',#'EMSR/HCR', 'EMSR/HAR',
				'HCR/OSA', 'HAR/OSA')
def scenas(scen, ratios):
	for r in ratios:
		newscen = copy(scen)
		newscen.f = (scen.f[0], scen.f[1]*r)
		newscen.nid = r
		yield newscen

pickle = 'fareratio.pkl'
gld = [s for s in scenas(sina, ((i/10.-.5)*.99999+.5 
		for i in range(1,10)))]
nosh.enuSim(gld, 10000, pickle, mymeth)
DISP = 'fareratio'
xlab = 'fare ratio'
nosh.drawSubFigs(DISP, xlab, *nosh.loadResults(pickle))
nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
