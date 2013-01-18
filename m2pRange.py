from noshow import *

vin = 5.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.3)
sina.f = (100.,40.0)
sina.V = 500.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (80.,90.)
sina.L = (40.,50.)
print "Load Factoer: ", sina.demandFactor(), sina.V
#sina.ddr = 1.1 #real demand rate 100%
#sina.pdr = 1.1 #real noshow rate 90%

sinb = copy(sina)
sinb.p = (0.15, 0.15)
print "Load Factoer: ", sinb.demandFactor()

sinb.nid = 0.0
sina.nid = 0.3
#glidePrint(sinb,sina,0,7,6.0)
glideSim(sinb, sina, 0, 7, 6.0, 10000, 'spread (mean fixed at 0.15)', 'prange')
#glideSim(sinb, sina, 0, 7, 6.0, 1000, 'spread (mean fixed at 0.15)')
