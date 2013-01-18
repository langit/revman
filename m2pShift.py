from noshow import *

vin = 5.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.2)
sina.f = (100.,40.0)
sina.V = 500.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (70.,80.)
sina.L = (40.,50.)
print "Demand Factor: ", sina.demandFactor(), "V=", sina.V

sinb = copy(sina)
sinb.p = (0.3, 0.5)
print "Demand Factor: ", sinb.demandFactor()

sinb.nid = 0.4
sina.nid = 0.1
#glidePrint(sina,sinb,0,7,6.0)
glideSim(sina, sinb, 0, 7, 6.0, 10000, 'mean no-show rate','pshift')
