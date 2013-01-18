from noshow import *
config.LBH=True
config.BINO=False
config.svl =0.001
mymeth = ['OBSA/CR','EMSR/CR','EMSR/NV', 
'CRSA/NV', 'DP/LBH', 'EMSR/SL','EMSR/NO']

vin = 5.0
sina = VixScena(500.0)
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.05,0.25)
sina.f = (100.,40.)
sina.V = 200.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (75.,85.)
sina.L = (45.,55.)
print "Load Factoer: ", sina.demandFactor(), sina.V
#sina.ddr = 1.1 #real demand rate 100%
#sina.pdr = 1.1 #real noshow rate 90%

sinb = copy(sina)
sinb.V = 800.0
print "Load Factoer: ", sinb.demandFactor()

sina.nid = sina.V
sinb.nid = sinb.V
