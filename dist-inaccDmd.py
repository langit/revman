from copy import copy
import pylab 
import noshow as nosh
#the demand bounds are wrong
#the true bounds will not change
class InaccDemand(nosh.SimScena):
    "for the ex3 in the OR paper: 2012/08/24"
    def __init__(me, realL, realU):
        me.ddr = 1.0#defDDR
        me.pdr = 1.0#defPDR
        me.realL = realL
        me.realU = realU
    def makeSource(me): #override
        src = nosh.UniformSource(me, me.ddr, me.pdr)
        src.L = me.realL
        src.U = me.realU
        return src

nosh.config.LBH=True
nosh.config.BINO=True
nosh.config.svl =0.001
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('EMSR/HCR', 'EMSR/HAR', 'EMSR/NV', 'DP/LBH', 
		   'EMSR/SL', 'EMSR/NO', 'HCR/OSA', 'HAR/OSA')
realL = (7., 25., 43., 10.)
realU = (27.,65., 103.,30.)
sina = InaccDemand(realL, realU)
sina.L = (12., 35., 58., 15.)
sina.U = (22., 55., 88., 25.)
sina.C = 124.0
sina.m = 4
sina.beta = 0.0
sina.p = (0.1, 0.1)
sina.f = (1050., 767., 527., 350.)
sina.V = 2000.0
#vin = 5.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
#sina.L = (50.,85.)
#sina.U = (70.,115.)
print "Demand Factor: ", sina.demandFactor(), "V=", sina.V

sinb = copy(sina)

sinb.L = (0., 5.,  13., 0.)
sinb.U = (34.,85., 133.,40.)
#sinb.L = (20.,40.)
#sinb.U = (100.,160.)
print "Demand Factor: ", sinb.demandFactor()

sina.nid = 0.5
sinb.nid = 2.0
xlab = 'demand interval ratio'

#nosh.glidePrint(sina,sinb,0,7,6.0)
pickle = 'dist-inaccDmd.pkl'
nosh.glideSim(sina, sinb, 0, 7, 6.0, 10000, xlab, pickle, mymeth)

DISP = None #'dist-inaccDmd'
nosh.drawFigs(DISP, xlab, *nosh.loadResults(pickle))
nosh.drawPolicies(DISP,xlab,*nosh.loadPolicies('policies-'+pickle))
if DISP is None: pylab.show()
