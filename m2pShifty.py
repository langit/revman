from noshow import *

vin = 5.0
sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.,0.3)
sina.f = (100.,40.0)
sina.V = sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.U = (70.,80.)
sina.L = (40.,50.)
print "Load Factoer: ", sina.getLoadFactor(), sina.V

sinb = copy(sina)
sinb.p = (0.3, 0.6)
###We adjust the bounds so that
###the effective bounds are constant
###**NOTE: effective bounds = bound * (1 - p)
###     where p is the average no-show rate
sinb.U = (105.,120)
sinb.L = (60.,75.)
print "Load Factoer: ", sinb.getLoadFactor()

sinb.nid = 0.45
sina.nid = 0.15

glideSim(sina, sinb, 0, 11, 10.0, 999, 'p: (.0,.3) --> (.3,.6)')
