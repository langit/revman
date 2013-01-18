from noshow import *

vin = 2.
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.2)
sina.f = (100.,50.0)
sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (120.,0.)
sina.L = (80.,0.)
print "Load Factoer: ", sina.getLoadFactor(), sina.V

sinb = copy(sina)
sinb.U = (0., 120.)
sinb.L = (0., 80.)
print "Load Factoer: ", sinb.getLoadFactor()

sina.nid = 1.
sinb.nid = 0.

glideSim(sinb, sina, 0, 11, 10.0, 700, 'Lo-to-Hi Mix')
