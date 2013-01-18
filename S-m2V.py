from noshow import *

vin = 4.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.3)
sina.f = (100.,40.0)
sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))
sina.U = (70.,80.)
sina.L = (40.,40.)
print "Load Factoer: ", sina.getLoadFactor(), sina.V

sinb = copy(sina)
#sinb.U = (40., 40., 70.)
#sinb.L = (20., 25., 40.)
sinb.V = sina.V*vin
print "Load Factoer: ", sinb.getLoadFactor()

sinb.nid = sinb.V
sina.nid = sina.V

glideSim(sina, sinb, 0, 11, 10.0, 999, 'Bumping Cost')
