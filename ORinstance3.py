from noshow import SimScena

sina = SimScena()
sina.L = (20., 40., 25.)
sina.U = (64.,150., 70.)
sina.C = 120.0
sina.m = 3
sina.beta = 0.1
sina.p = (0.1, 0.1)
sina.f = (1050., 647., 350.)
sina.V = 2500.0 #1500.0

print "Load Factor: ", sina.demandFactor(), "V=", sina.V
