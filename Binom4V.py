from noshow import *
config.BINO=True
vin = 4.0
sina = SimScena()
sina.C = 124.0
sina.m = 4
sina.beta = 0.2
sina.p = (0.0,0.2)
#example adopted from Talluri&van Ryzin (2004a)
#Section 2.2.3.4
sina.f = (1050.,567.,527.,350.)
sina.V = 1.2*sina.f[0]/(1.0 - sum(sina.p)/2.0)
#mean demand: 17.3, 45.1, 73.6, 19.8
mean =(17.3, 45.1, 73.6, 19.8)
#demand sigma: 5.8, 15.0, 13.2, 11.3
stdev = (5.8, 15.0, 13.2, 11.3)
sina.U = [int(m+2*s) for m,s in zip(mean, stdev)]
sina.L = [max(0, int(m-2*s) for m,s in zip(mean, stdev)]
print sina.L
print sina.U
print "Load Factoer: ", sina.demandFactor(), sina.V

sinb = copy(sina)
#sinb.U = (40., 40., 70.)
#sinb.L = (20., 25., 40.)
sinb.V *= vin
print "Load Factoer: ", sinb.demandFactor()
sina.nid = sina.V
sinb.nid = sinb.V
#glidePrint(sina,sinb, 0, 6, 5.0)
#glideSim(sina, sinb, 0, 6, 5.0, 10000, 'overbooking cost (V): m=4', 'binom4v', ('EMSR/CR', 'OBSA/CR', 'EMSR/NV', 'EMSR/SL', 'CRE/OSA'))
glideSim(sina, sinb, 0, 6, 5.0, 5000, 'overbooking cost (V): m=4', None, ('EMSR/CR', 'OBSA/CR', 'EMSR/NV', 'EMSR/SL', 'CRE/OSA'))
