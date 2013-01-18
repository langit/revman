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
sina.U = (105.,120.)
sina.L = (60.,75.)
print "Demand Factor: ", sina.demandFactor(), "V=", sina.V

sinb = copy(sina)
sinb.U = (49.,56.)
sinb.L = (28.,35.)
print "Demand Factor: ", sinb.demandFactor()

sinb.nid = sinb.demandFactor()
sina.nid = sina.demandFactor()

#glidePrint(sinb, sina, 0, 7, 6.0)
glideSim(sina, sinb, 0, 7, 6.0, 10000, 'demand factor', 'dfact')
