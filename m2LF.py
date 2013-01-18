from noshow import *

vin = 4.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.2)
sina.f = (100.,50.0)
sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (80.,40.)
sina.L = (30.,20.)
print "Load Factoer: ", sina.getLoadFactor(), sina.V

sinb = copy(sina)
sinb.U = (208., 104.)
sinb.L = (80., 52.)
print "Load Factoer: ", sinb.getLoadFactor()

sinb.nid = sinb.getLoadFactor()
sina.nid = sina.getLoadFactor()

glideSim(sina, sinb, 0, 11, 10.0, 999, 'Load Factor')
