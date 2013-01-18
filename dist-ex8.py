from noshow import *
config.LBH=True
config.BINO=True
config.svl =0.001
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL','EMSR/NO', 'CRE/OSA')

class TriangleSource(UniformSource):
    def __init__(me, scen, ddr, pdr):
        me.m = len(scen.U)
        me.U = [scen.U[i]/ddr for i in range(me.m)]
        me.L = [scen.L[i]/ddr for i in range(me.m)]
        me.p = [scen.p[i]/pdr for i in range(2)]

    def nextProfile(me, cor):
        return [int(me.triangle(i, True)) for i in range(me.m)]

class myScena(SimScena):
    def __init__(me):
        me.ddr = defDDR
        me.pdr = defPDR
        if not config.BINO:
           print "BinoScena WARNING: set config.BINO to True"

    def makeSource(me):
        return TriangleSource(me, me.ddr, me.pdr)

sina = myScena()
sina.C = 124.0
sina.m = 3
sina.beta = 0.2
sina.p = (0.0,0.2)
#example adopted from Talluri&van Ryzin (2004a)
#Section 2.2.3.4, but merged the middle classes
sina.f = (1050.,647.,350.)
sina.V = 3000.0
#mean demand: 17.3, 45.1, 73.6, 19.8
sina.U = (40.,80.,100.)
sina.L = (15.,30. ,10.)
print "Load Factoer: ", sina.demandFactor(), sina.V

src = sina.makeSource()
avg = sum([src.triangle(1, True) for i in range(200)])/200.
print "average:", avg, "bound:", sina.L[1], sina.U[1] 



sinb = copy(sina)
print "Load Factoer:", sinb.demandFactor()
sinb.V = 6000.
sina.nid = sina.V
sinb.nid = sinb.V

#glidePrint(sina,sinb, 0, 6, 5.0)
glideSim(sina, sinb, 0, 6, 5.0, 10000, 'Denial Cost (m=3)', 'dist-ex8-derror', mymeth)
#glideSim(sina, sinb, 0, 5, 6.0, 10000, 'overbooking cost (V): m=3')
