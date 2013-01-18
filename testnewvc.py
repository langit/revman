from noshow import *

vin = 2.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.2)
sina.f = (100.,40.0)
sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (70.,80.)
sina.L = (40.,50.)
print "Load Factoer: ", sina.getLoadFactor(), sina.V
VC = sina.newsVC(vbs=True)
print 'VC=', VC

print "Now Lan's Version:"
VC = sina.newsVCL(vbs=True)
print 'VC=', VC
