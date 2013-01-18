from noshow import *

vin = 4.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.2)
sina.f = (100.,40.0)
sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))
sina.U = (70.,80.)
sina.L = (40.,50.)
print "Load Factoer: ", sina.getLoadFactor(), sina.V

reps = 2
COR = False
LBH = False
ptype = 2
BINO = True
scenaSim(sina, reps, COR, LBH, ptype, BINO)
