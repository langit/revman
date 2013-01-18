from noshow import *

vin = 4.0
sina = SimScena()
sina.C = 124.0
sina.m = 4
sina.beta = 0.2
sina.p = (0.0,0.1)
#example adopted from Talluri&van Ryzin (2004a)
#Section 2.2.3.4
sina.f = (1050.,567.,527.,350.)
sina.V = 5000.
meanD = (17.3, 45.1, 73.6, 19.8)
sina.U = tuple([1.3*meanD[i] for i in range(4)])
#sina.L = (0.,0.,0.,0.)
sina.L = tuple([.3*meanD[i] for i in range(4)])
print "Load Factoer: ", sina.demandFactor(), sina.V

sinb = copy(sina)
sinb.U = tuple([2.*sina.U[i] for i in range(4)])
sinb.L = tuple([2.*sina.L[i] for i in range(4)])
print "Load Factoer: ", sinb.demandFactor()
sina.nid = sina.demandFactor()
sinb.nid = sinb.demandFactor()
#glidePrint(sina,sinb, 0, 6, 5.0)
glideSim(sina, sinb, 0, 6, 5.0, 10000, 'demand factor: m=4', 'm4dfact')
#glideSim(sina, sinb, 0, 6, 5.0, 5000, 'demand factor: m=4')
