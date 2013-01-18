from noshow import *
config.LBH=True
config.BINO=False
config.svl =0.001
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ['OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL','EMSR/NO']

vin = 5.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.2)
sina.f = (500.,100.)
sina.V = 900.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (80.,80.)
sina.L = (40.,40.)
print "Demand Factor: ", sina.demandFactor(), "V=", sina.V

sinb = copy(sina)
sinb.p = (0.3, 0.5)
print "Demand Factor: ", sinb.demandFactor()

sinb.nid = 0.4
sina.nid = 0.1
xlab = 'mean no-show rate'

#glidePrint(sina,sinb,0,7,6.0)
#glideSim(sina, sinb, 0, 7, 6.0, 9000, xlab,'msom-ex2-pshift.pkl', mymeth)
glide = ScenaGlide(sina, sinb)
gmid = glide.getGlide(0.5)
print "Nid:", gmid.nid
#scenaQuant(gmid, mymeth, 'msom-ex2ms-pshift')
scenaQuant(gmid, mymeth, None, True)
