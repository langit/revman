'''
the actual demand is normal
'''

import noshow as nosh
from random import uniform, normalvariate as vnorm
from copy import copy
import pylab
from normalcdf import phinv, phi

nosh.config.LBH=True
nosh.config.BINO=True
nosh.config.svl =0.001

nosh.config.samples = 3000
nosh.config.elites  = 300 
nosh.config.smoother = 0.7

#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('DP/LBH', 'EMSR/NV', 'EMSR/SL', 'HCR/OSA', 'HAR/OSA')

class NormSrc(nosh.UniformSource):
    def __init__(me, scen):
        me.m = len(scen.U)
        me.p = scen.p
        me.mu = scen.mu
        me.sigma = scen.sigma

    def nextProfile(me, cor):
        assert not cor, "unable to process correlated demand."
        d=[int(vnorm(me.mu[i], me.sigma[i])) for i in range(me.m)]
        return [max(0,di) for di in d]

    #Below compute the exact static policy for LBH arrivals
    #assuming demand is discrete normal.
    #noshow rate is also uniform (continous).
    #or noshow is binomial with uniform probability of noshow
    #In both cases, the termianl cost is CONVEX
    def policyStaticDP(me): #ref. van Ryzin RM book pg 157
        #first test for the uppper bound of overbooking.
        V = 1.0 - sum(me.p)/2.0
        f,L,U = (0,)+me.f,(0,)+me.L,(0,)+me.U
        if V*me.V <= f[1]:
            raise "static model violates COR! %f"%(f[1]/V)
        V = None #variables for Bellman equaiton
        B = 2+int(me.C / (1.0 - me.p[1])) #?
        if config.BINO: #if binomial distribution
            pf = penub(me.p,me.V,me.C,B,penadj(0,0,0,0))
            V = [-pf.g(i) for i in range(B)]
            #print V[int(me.C):]
        else:
            V = [0.0 for i in range(B)]
            V[B-1]=-me.V*((1.0-sum(me.p)/2.0)*(B-1)-me.C)
            #pf = penun(me.p, me.V, me.C,penadj(0,0,0,0))
            for i in range(1+int(me.C/(1.0-me.p[0])), B-1):
                V[i] = (i - i * me.p[0] - me.C)
                V[i] *= - V[i] * me.V
                V[i] /= float(2*i) * (me.p[1] - me.p[0])
                #print V[i] + pf.g(i), #pf for debug

        b = [0 for i in range(me.m+1)] #policy
        b[0] = len(V) - 1
        for j in range(1,me.m+1):
            b[j] = b[j-1]
            K = float(U[j]-L[j]+1)
            ll, uu = int(L[j]), int(U[j])
            while f[j] < V[b[j]-1] - V[b[j]]: 
                b[j] -= 1
                if b[j] < 1: break 
            if b[j] < 1: break
            #update V: Vj(y) = E[max_{u<=Dj} f_j u + V_{j-1}(y+u)]
			# if y>=b_j: V_{j-1}(y) - V_{j-1}(y-1) < f_j: 
            #   argmax_{u<=Dj} f_j u + V_{j-1}(y+u) = 0
            # therefore, V_j(y) = V_{j-1}(y)
			# if y<b_j: V_{j-1}(y) - V_{j-1}(y-1)>f_j: 
			#   argmax_{u<=Dj} f_j u + V_{j-1}(y+u) = min(Dj,b_j-y)
			#   if D_j >= b_j - y: u* = b_j-y
			#     f_j (b_j-y) + V_{j-1}(b_j)
			#   if D_j < b_j - y: u* = Dj
			#     f_j D_j + V_{j-1}(b_j+Dj)
			# therefore, V_j(y) = P{Dj>=b_j-y} 
			#                     (f_j (b_j-y) + V_{j-1}(b_j)+
			#      sum{P{D_j=d} [f_j d + V_{j-1}(y+d)]; d<b_j-y}
            W = V[:] #case y >= b_j
            for y in range(b[j] - 1, -1, -1):
                t = max(b[j] - y, 0) 
                St = 1. - phi(t) if t else 1. 
                pt = [phi(d) for d in range(t)]
                pt = [pt[d]-pt[d-1] if d else pt[d] 
								for d in range(t)]
                lt = sum(pt[d]*(f[j]*d + V[y+d]) 
								for d in range(t))
                W[y] = St*V[y] + lt
            V = W
        assert b[1] < b[0], "B is not big enough!"
        for i in range(1, me.m):
            b[i] = b[i]-b[i+1]
        return b

class NormalScena(nosh.SimScena):
    '''converts an uniform demand distr. scen into normal demand'''
    def __init__(me, sina):
        me.L = sina.L
        me.U = sina.U
        me.C = sina.C
        me.m = sina.m
        me.p = sina.p
        me.f = sina.f
        me.beta = sina.beta
        me.V = sina.V
        me.mu = [(l+u)/2. for l,u in zip(me.L,me.U)]
        me.sigma = [(u-l)/12**.5 for l,u in zip(me.L,me.U)]

    def makeSource(me): #override
        return NormSrc(me)

    #below we overide this to have correct EMSR
    def invSurvival(me, p, i): #inverse Survival
        return me.mu[i-1] + me.sigma[i-1]*phinv(1-p)

    #TODO: still need to rewrite DP/LBH

def scenas(scen, ps):
	for C in ps:
		newscen = copy(scen)
		newscen.C = newscen.nid = C 
		print "## LOAD Factor:", newscen.demandFactor()
		yield newscen

from ORinstance3 import sina as osin
sina = NormalScena(osin)

pickle = 'df3normal.pkl'
gld = [s for s in scenas(sina, range(100, 201, 10))]
nosh.enuSim(gld, 10000, pickle, mymeth)
DISP = 'ORdf3normal'
xlab = 'total number of seats'
nosh.drawSubFigs(DISP, xlab, *nosh.loadResults(pickle))
nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies(pickle))
if DISP is None: pylab.show()
