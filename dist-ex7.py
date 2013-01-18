from noshow import *
config.LBH=True
config.BINO=True
config.svl =0.001
config.confid =0.05
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV','CRSA/NV','DP/LBH','EMSR/S1')
mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL','EMSR/NO', 'CRE/OSA')

vin = 5.0
sina = BinoScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.3)
sina.f = (100.,40.)
sina.V = 500.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (70.,80.)
sina.L = (40.,50.)
print "Load Factoer: ", sina.demandFactor(), sina.V
#sina.ddr = 1.1 #real demand rate 100%
#sina.pdr = 1.1 #real noshow rate 90%

sinb = copy(sina)
sinb.p = (0.15, 0.15)
print "Load Factoer: ", sinb.demandFactor()

sinb.nid = 0.0
sina.nid = 0.3

print sina.p
print sina.makeProblem().p

#glidePrint(sinb,sina,0,7,6.0)
glideSim(sinb, sina, 0, 7, 6.0, 9000, 'spread (mean fixed at 0.15)', 'dist-ex7-prange', mymeth)
#glideSim(sinb, sina, 0, 7, 6.0, 1000, 'spread (mean fixed at 0.15)')
