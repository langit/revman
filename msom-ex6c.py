from noshow import *
config.LBH=True
config.BINO=True
config.svl =0.001
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL','EMSR/NO')

vin = 5.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.2)
sina.f = (100.,40.)
sina.V = 500.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (70.,80.)
sina.L = (40.,50.)
print "Demand Factor: ", sina.demandFactor(), "V=", sina.V

sinb = copy(sina)
sinb.p = (0.3, 0.5)
print "Demand Factor: ", sinb.demandFactor()
sinb.nid = 0.4
sina.nid = 0.1

glide = ScenaGlide(sina, sinb)
vvv = [k/6.0 for k in range(0, 7)]
gld = [glide.getGlide(v) for v in vvv]
stq = 1.-sum(gld[0].p)/2.
stL = gld[0].L
stU = gld[0].U
stm = gld[0].m
for ss in gld:
    if ss == gld[0]: continue
    ff = stq/(1.-sum(ss.p)/2.)
    ss.L = tuple([stL[i]*ff for i in range(stm)])
    ss.U = tuple([stU[i]*ff for i in range(stm)])
    print ss.demandFactor(),
print

#glidePrint(sina,sinb,0,7,6.0)
enuSim(gld, 9000, 'mean no-show rate','msom-ex6-fixDF', mymeth)
