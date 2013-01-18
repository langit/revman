from noshow import *
config.LBH=True
config.BINO=False
config.svl =0.001
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL','EMSR/NO', 'CRE/OSA')

vin = 5.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.1,0.2)
sina.f = (100.,0.01)
sina.V = 500.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (70.,80.)
sina.L = (40.,50.)
print "Demand Factoer: ", sina.demandFactor(), sina.V

sinb = copy(sina)
sinb.f = (100., 99.99)
print "Demand Factoer: ", sinb.demandFactor()
sinb.nid = 100. 
sina.nid = 0.0

#glidePrint(sina,sinb,0,6,5.0)
glideSim(sina, sinb, 0, 6, 5.0, 9000, r'$f_2$', 'dist-ex5-fratio',mymeth)
