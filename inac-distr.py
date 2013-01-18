'''
all models assume uniform demand distr.
but the actual distr. is a mixtrure of
two triangle distr. -- the left triangle
and the right triangle. the mixture 
probability is p: with chance of p, the
sample comes from the left triangle,
and with (1-p) it comes from the right
triangle. if p=0.5, it is uniform!
we use the nid field of SimScena for
the mixture probability p.
'''

import noshow as nosh
from random import uniform
import pylab 
from copy import copy

opposfares = 2 #fare-class 1 and 2 are opposite

class MixedTriangles(nosh.UniformSource):
    def __init__(me, scen):
        nosh.UniformSource.__init__(me, 
				scen, scen.ddr, scen.pdr)
        me.nid = scen.nid
        me.oppos = opposfares

    def nextProfile(me, cor=None):
        prof = [None for k in range(me.m)]
        for k in range(me.m):
            Left = (uniform(0,1) > me.nid)
            if k < me.oppos: Left = not Left
            prof[k] = int(me.triangle(k, Left))
        return prof

class InacDist(nosh.SimScena):
    def makeSource(me):
        return MixedTriangles(me)

nosh.config.LBH=True
nosh.config.BINO=True
nosh.config.svl =0.001

nosh.config.samples = 1500
nosh.config.elites  = 90
nosh.config.smoother = 0.6

#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('EMSR/HCR', 'EMSR/HAR', 'EMSR/NV', 'DP/LBH', 
		   'EMSR/SL', 'EMSR/NO', 'HCR/OSA', 'HAR/OSA')
sina = InacDist()
sina.L = (7., 25., 43., 10.)
sina.U = (27.,65., 103.,30.)
sina.C = 124.0
sina.m = 4
sina.beta = 0.1
sina.p = (0.1, 0.1)
sina.f = (1050., 767., 527., 350.)
sina.V = 2500.0
#vin = 5.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
#sina.L = (50.,85.)
#sina.U = (70.,115.)
print "Demand Factor: ", sina.demandFactor(), "V=", sina.V

sinb = copy(sina)
sina.nid = 0.0
sinb.nid = 1.0
xlab = 'mixture probability'

#nosh.glidePrint(sina,sinb,0,7,6.0)
pickle = 'inac-distr.pkl'

#nosh.glideSim(sina, sinb, 0, 11, 10.0, 10000, pickle, mymeth)
DISP = None#'inac-dist'

nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
#nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
