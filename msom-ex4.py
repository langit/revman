from noshow import *
config.LBH=True
config.BINO=False
config.svl =0.001
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL','EMSR/NO')

vin = 3.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.05, 0.15)
sina.f = (100.,40.0)
#sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))
#print sina.V,
sina.V = 200.0 #1.5*sina.f[0]/(1.0 - sum(sina.p)/2.0)
sina.U = (75.,85.)
sina.L = (45.,55.)

sinb = copy(sina)
sinb.V = 600.0 #sina.V*vin

sinb.nid = sinb.V 
sina.nid = sina.V 

#glidePrint(sina,sinb, 0, 6, 5.0)
glideSim(sina, sinb, 0, 6, 5.0, 9000, 'overbooking cost (V)', 'msom-ex4-bcost',mymeth)
#glideSim(sina, sinb, 0, 6, 5.0, 5000, 'overbooking cost (V)')
