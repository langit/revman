'''
the effect of uncertainty set.
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

#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('EMSR/NV', 'DP/LBH', 'EMSR/SL', 'HCR/OSA', 'HAR/OSA')

def scenas(scen, ps):
	for us in ps:
		newscen = copy(scen)
		newscen.uset = newscen.nid = us
		print "## Scenario LOAD Factor:", newscen.demandFactor()
		yield newscen

from ORinstance3 import sina

pickle = 'uset3.pkl'
gld = [s for s in scenas(sina, (i/10. for i in range(11)))]
nosh.enuSim(gld, 10000, pickle, mymeth)
DISP = 'ORuset3'
xlab = 'unsertainty set factor'
nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
#nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
