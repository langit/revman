'''
Try to measure in multiple ways.
'''

import noshow as nosh
from random import uniform, normalvariate as vnorm
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

class NormSrc(nosh.UniformSource):
    def __init__(me, scen, ddr, pdr):
        me.m = len(scen.U)
        me.U = [scen.U[i]/ddr for i in range(me.m)]
        me.L = [scen.L[i]/ddr for i in range(me.m)]
        me.p = [scen.p[i]/pdr for i in range(2)]
        me.mu = [(scen.L[i]+scen.U[i])/2. for i in range(me.m)]
        me.sigma = [(scen.U[i]-scen.L[i])/12**.5 for i in range(me.m)]

    def nextProfile(me, cor):
        assert not cor, "unable to process correlated demand."
        d=[int(vnorm(me.mu[i], me.sigma[i])) for i in range(me.m)]
        return [max(0,di) for di in d]

class NormScena(nosh.SimScena):
    def makeSource(me): #override
        return NormSrc(me, me.ddr, me.pdr)

def scenas(scen, ps):
	for C in ps:
		newscen = copy(scen)
		newscen.C = newscen.nid = C 
		print "## LOAD Factor:", newscen.demandFactor()
		yield newscen

from ORinstance3 import sina as osin
sina = NormScena()
sina.L = osin.L
sina.U = osin.U
sina.C = osin.C
sina.m = osin.m
sina.p = osin.p
sina.f = osin.f
sina.beta = osin.beta
sina.V = osin.V

pickle = 'normdf3.pkl'
gld = [s for s in scenas(sina, range(100, 201, 10))]
nosh.enuSim(gld, 10000, pickle, mymeth)
DISP = 'ORnormdf3'
xlab = 'total number of seats'
nosh.drawSubFigs(DISP, xlab, *nosh.loadResults(pickle))
nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
#nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
