from noshow import *

vin = 4.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.1,0.2)
sina.f = (100.,50.)
sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (280.,0.)
sina.L = (20.,0.)
print "Load Factoer: ", sina.getLoadFactor(), sina.V

sinb = copy(sina)
sinb.U = (0., 180.)
sinb.L = (0., 120.)
print "Load Factoer: ", sinb.getLoadFactor()

sina.nid = 1.
sinb.nid = 0.

glideSim(sinb, sina, 0, 11, 10.0, 999, 'All Lo --> All Hi')
