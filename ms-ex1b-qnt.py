from noshow import *
config.LBH=True
config.BINO=False
config.svl =0.001
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('OBSA/CR', 'OBSA/AR', 'EMSR/NO', 'Off/Opt' )

sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.05, 0.15)
sina.f = (500.,100.)
sina.V = 700.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (80.,80.)
sina.L = (40.,40.)
print "Demand Factor: ", sina.demandFactor(), "V=", sina.V

scenaQuant(sina, mymeth, 'ms-ex1b')
