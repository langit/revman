from noshow import *

vin = 4.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.0,0.2)
sina.f = (100.,40.0)
sina.V = 400.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (70.,80.)
sina.L = (40.,50.)
print "Load Factoer: ", sina.demandFactor(), sina.V
sina.ddr = 0.8 #real noshow rate 90%

sinb = copy(sina)
sinb.ddr = 1.2
print "Load Factoer: ", sinb.demandFactor()

sinb.nid = (sinb.ddr - 1.0) * 100.
sina.nid = (sina.ddr - 1.0) * 100.
#glidePrint(sinb,sina,0,7,6.0)
glideSim(sinb, sina, 0, 7, 6.0, 10000, 'demand bounds error %', 'derr')
#glideSim(sinb, sina, 0, 7, 6.0, 3000, 'demand bounds error %')
