from noshow import SimScena, NormalScena

sina = SimScena()
#(7, 19, 50, 0) (27, 71, 96, 39)
sina.L = (15., 40., 25.)
sina.U = (64.,150., 70.)
#ORinstHow.py: 
#sina.L, sina.U = (8., 94., 0.),(30., 170., 43.)
sina.C = 125.0
sina.m = 3
sina.beta = 0.1
sina.p = (0.1, 0.1)
sina.f = (1050., 647., 350.)
sina.V = 2500. #1500.0
print "Load Factor: ", sina.demandFactor(), "V=", sina.V
NormalScena(sina)
