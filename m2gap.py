from noshow import *

vin = 2.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.2)
sina.f = (100.,50.0)
sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (60.,80.)
sina.L = (50.,70.)
print "Load Factoer: ", sina.getLoadFactor(), sina.V

sinb = copy(sina)
sinb.U = (80., 100.)
sinb.L = (30., 50.)
print "Load Factoer: ", sinb.getLoadFactor()

sinb.nid = 40.
sina.nid = 10.

glideSim(sina, sinb, 0, 11, 10.0, 700, 'Gap Index')
