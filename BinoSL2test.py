from noshow import *
config.BINO=True
config.svl = 0.01

vin = 4.0
sina = SimScena()
sina.C = 124.0
sina.m = 3
sina.beta = 0.2
sina.p = (0.0,0.2)
#example adopted from Talluri&van Ryzin (2004a)
#Section 2.2.3.4, but merged the middle classes
sina.f = (1050.,647.,350.)
sina.V = 1.2*sina.f[0]/(1.0 - sum(sina.p)/2.0)
#mean demand: 17.3, 45.1, 73.6, 19.8
sina.U = (44.,107.,49.)
sina.L = (10.,30. ,10.)
print "Load Factoer: ", sina.demandFactor(), sina.V

sinb = copy(sina)
#sinb.U = (40., 40., 70.)
#sinb.L = (20., 25., 40.)
sinb.V *= vin
print "Load Factoer: ", sinb.demandFactor()
sina.nid = sina.V
sinb.nid = sinb.V
#glidePrint(sina,sinb, 0, 6, 5.0)
#glideSim(sina, sinb, 0, 6, 5.0, 10000, 'overbooking cost (V): m=3', 'm3bcost')
#glideSim(sina, sinb, 0, 5, 6.0, 10000, 'overbooking cost (V): m=3')
scenaPrint(sina)
