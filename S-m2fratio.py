from noshow import *

vin = 5.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.2)
sina.f = (100.,0.01)
sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (70.,80.)
sina.L = (40.,50.)
print "Load Factoer: ", sina.getLoadFactor(), sina.V

sinb = copy(sina)
sinb.f = (100., 99.99)
print "Load Factoer: ", sinb.getLoadFactor()

sinb.nid = 1.0
sina.nid = 0.0

glideSim(sina, sinb, 0, 11, 10.0, 999, 'Fare Ratio')
