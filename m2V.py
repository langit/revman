from noshow import *

vin = 3.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.3)
sina.f = (100.,40.0)
#sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))
#print sina.V,
sina.V = 200.0 #1.5*sina.f[0]/(1.0 - sum(sina.p)/2.0)
print sina.V
sina.U = (70.,80.)
sina.L = (40.,50.)

sinb = copy(sina)
sinb.V = 700.0 #sina.V*vin

sinb.nid = sinb.V 
sina.nid = sina.V 

print sina.V, sinb.V
#glidePrint(sina,sinb, 0, 6, 5.0)
glideSim(sina, sinb, 0, 6, 5.0, 10000, 'overbooking cost (V)', 'bcost')
#glideSim(sina, sinb, 0, 6, 5.0, 5000, 'overbooking cost (V)')
