from noshow import *
config.BINO = True

vin = 5.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.3)
sina.f = (100.,40.0)
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
#glidePrint(sinb,sina,0,7,6.0)

#to make the bino files: methods also listed.
#glideSim(sinb, sina, 0, 7, 6.0, 10000, 'spread (mean fixed at 0.15)', None, ('EMSR/CR', 'OBSA/CR', 'EMSR/NV', 'EMSR/SL', 'CRE/OSA'))
glideSim(sinb, sina, 0, 7, 6.0, 10000, 'spread (mean fixed at 0.15)', 'prangeBino', ('EMSR/CR', 'OBSA/CR', 'EMSR/NV', 'EMSR/SL', 'CRE/OSA', 'DP/LBH'))
