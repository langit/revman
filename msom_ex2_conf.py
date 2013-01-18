from noshow import *
config.LBH=True
config.BINO=False
config.svl =0.001
mymeth = ['OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL','EMSR/NO']

vin = 5.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.2)
sina.f = (100.,40.)
sina.V = 500.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (75.,85.)
sina.L = (45.,55.)
print "Demand Factor: ", sina.demandFactor(), "V=", sina.V

sinb = copy(sina)
sinb.p = (0.1, 0.3)
print "Demand Factor: ", sinb.demandFactor()

sinb.nid = 0.19999999
sina.nid = 0.1
