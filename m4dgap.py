from noshow import *

vin = 4.0
sina = SimScena()
sina.C = 124.0
sina.m = 4
sina.beta = 0.2
sina.p = (0.0,0.2)
sina.f = (1050.,567.,527.,350.)
sina.V = 5000.
meanD = (17.3, 45.1, 73.6, 19.8)
sina.U = tuple([2*meanD[i] for i in range(4)])
sina.L = (0.,0.,0.,0.)
print "Load Factoer: ", sina.demandFactor(), sina.V

sinb = copy(sina)
sinb.U = meanD
sinb.L = meanD
print "Load Factoer: ", sinb.demandFactor()
sina.nid = 2.0
sinb.nid = 0.0
#glidePrint(sina,sinb, 0, 6, 5.0)
glideSim(sina, sinb, 0, 6, 5.0, 5000, 'demand gap/mean ratio: m=4', 'm4dgap')
#glideSim(sina, sinb, 0, 6, 5.0, 5000, 'overbooking cost (V): m=4')
