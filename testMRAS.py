from noshow import *

def testit(prob):
    nexp = noshexp(prob, config.tadj)
    std = [max(1.0,(prob.U[i]-prob.L[i])/2.0) for i in range(0,prob.m+1)]
    avg = [(prob.L[i]+prob.U[i])/2.0 for i in range(0,prob.m+1)]
    #u, s, z, e, alpha
    nexp.initMRAS(avg, std, 200, 10, 0.4 )
    nexp.search(21)
    std = [max(1.0,(prob.U[i]-prob.L[i])/2.0) for i in range(0,prob.m+1)]
    avg = [(prob.L[i]+prob.U[i])/2.0 for i in range(0,prob.m+1)]
    nexp.initMRAS(avg, std, 200, 10, 0.4 )
    nexp.search(21)

vin = 2.0
sina = SimScena()
sina.C = 124.0
sina.m = 3
sina.beta = 0.2
sina.p = (0.0,0.2)
#example adopted from Talluri&van Ryzin (2004a)
#Section 2.2.3.4, but merged the middle classes
sina.f = (1050.,547.,350.)
sina.V = vin*sina.f[0]/(1.0 - sum(sina.p)/2.0)
#mean demand: 17.3, 45.1, 73.6, 19.8
sina.U = (34.,197.,39.)
sina.L = (9.,20.,1.)

testit(sina.makeProblem())
