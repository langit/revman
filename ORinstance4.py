from noshow import *
sina = SimScena()
sina.C = 110.0
sina.m = 4
sina.beta = 0.1
sina.p = (0.1,0.1)
#example adopted from Talluri&van Ryzin (2004a)
#Section 2.2.3.4
sina.f = (1050.,567.,527.,350.)
sina.V = 2500.0 #1.2*sina.f[0]/(1.0 - sum(sina.p)/2.0)
#mean demand: 17.3, 45.1, 73.6, 19.8
mean =(17.3, 45.1, 73.6, 19.8)
#demand sigma: 5.8, 15.0, 13.2, 11.3
stdev = (5.8, 15.0, 13.2, 11.3)
sina.U = tuple(int(m+2*s) for m,s in zip(mean, stdev))
sina.L = tuple(max(0, int(m-2*s)) for m,s in zip(mean, stdev))
print sina.L, sina.U
print "LoadFactor:", sina.demandFactor(), "V=", sina.V
