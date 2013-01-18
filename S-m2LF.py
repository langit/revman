from noshow import *

vin = 4.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.1,0.3)
sina.f = (100.,60.0)
sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (34.,70.)
sina.L = (25.,25.)
print "Load Factoer: ", sina.getLoadFactor(), sina.V

#LF = 1.0:
#(55, 115)
#(40, 40)

sinb = copy(sina)
sinb.U = (99., 198.)
sinb.L = (74., 74.)
print "Load Factoer: ", sinb.getLoadFactor()

sinb.nid = sinb.getLoadFactor()
sina.nid = sina.getLoadFactor()

glideSim(sina, sinb, 0, 11, 10.0, 999, 'Load Factor')
