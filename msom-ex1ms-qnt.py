from noshow import *
config.LBH=True
config.BINO=False
config.svl =0.001
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL','EMSR/NO')
mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL','EMSR/NO')

sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.2)
sina.f = (500.,100.)
sina.V = 700.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (90.,90.)
sina.L = (60.,60.)
print "Load Factoer: ", sina.demandFactor(), sina.V
#sina.ddr = 1.1 #real demand rate 100%
#sina.pdr = 1.1 #real noshow rate 90%

sinb = copy(sina)
sinb.p = (0.1, 0.1)
print "Load Factoer: ", sinb.demandFactor()

sinb.nid = 0.0
sina.nid = 0.2
glide = ScenaGlide(sinb, sina)
gmid = glide.getGlide(0.2)
print "Nid (Spread):", gmid.nid
scenaQuant(gmid, mymeth, 'msom-ex1ms-prange')

#glidePrint(sinb,sina,0,7,6.0)
#glideSim(sinb, sina, 0, 7, 6.0, 9000, 'spread (mean fixed at 0.15)', 'msom-ex1-prange', mymeth)
#glideSim(sinb, sina, 0, 7, 6.0, 1000, 'spread (mean fixed at 0.15)')
