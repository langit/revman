import noshow as nosh
from random import uniform
from copy import copy
import pylab

from ORinstance3 import sina
import sys
if 'normal' == sys.argv[-1]:
	sina = nosh.NormalScena(sina)

nosh.config.LBH=True
nosh.config.BINO=True
nosh.config.svl =0.001

nosh.config.samples = 3000 #2900
nosh.config.elites  = 300 #200 
nosh.config.smoother = 0.5
nosh.config.iters = 51

mymeth = ('DP/LBH', 'EMSR/NV', 'EMSR/SL',#'EMSR/HCR', 'EMSR/HAR',
				'HCR/OSA', 'HAR/OSA')
def scenas(scen, ratios):
	f = scen.f
	rat = f[2]/float(f[1])
	for r in ratios:
		newscen = copy(scen)
		f1 = f[0]*r 
		newscen.f = (f[0], f1, f1*rat)
		newscen.nid = r
		yield newscen

pickle = 'fareratio3.pkl'
gld = [s for s in scenas(sina, (i/10. for i in range(3,8)))]
#nosh.enuSim(gld, 10000, pickle, mymeth)
DISP = 'ORfareratio3'
xlab = 'fare ratio'
nosh.drawSubFigs(DISP, xlab, *nosh.loadResults(pickle), no_percentiles=True)
#nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
#nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
