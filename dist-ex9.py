from noshow import *
config.LBH=True
config.BINO=False
config.svl =0.001
mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL','EMSR/NO', 'CRE/OSA')

vin = 5.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.1,0.2)
sina.f = (100.,40.)
sina.V = 500.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (160.,0.)
sina.L = (80.,0.)
print "Demand Factoer: ", sina.demandFactor(), sina.V

sinb = copy(sina)
sinb.U = (0., 140.)
sinb.L = (0., 100.)
print "Demand Factoer: ", sinb.demandFactor()

sina.nid = 1.
sinb.nid = 0.
#glidePrint(sinb, sina, 0, 6, 5.0)
glideSim(sinb, sina, 0, 6, 5.0, 8000, r'$\alpha$', 'dist-ex9-dmix', mymeth)
