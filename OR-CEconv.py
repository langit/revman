from noshow import *
config.BINO = True
config.tadj = 0
#config.svlp = 200.0

config.vsearch = 1 #print each iteration
config.samples = 1600
config.elites  = 80 
config.smoother = 0.7

from ORinstance3 import sina
#example adopted from Talluri&van Ryzin (2004a)
#Section 2.2.3.4, but merged the middle classes
#sina = SimScena()
#sina.C = 108.0
#sina.m = 3
#sina.beta = 0.2
#sina.p = (0.1,0.1)
#sina.f = (1050.,647.,350.)
#sina.V = 1.2*sina.f[0]/(1.0 - sum(sina.p)/2.0)
#mean demand: 17.3, 45.1, 73.6, 19.8
#sina.U = (64.,120.,39.)
#sina.L = (20., 30., 0.)
#print "Demand Factoer:", sina.demandFactor(), sina.V

nexp = noshexp(sina, 0)
prob = sina.makeProblem()

utot = int(sum(sina.U))
ltot = int(sum(sina.L))

sump = [k for k in range(ltot, utot+1)]
#std = [max(1.0,(prob.U[i]-prob.L[i])/2.0) for i in range(0,prob.m+1)]
std = [prob.U[i] for i in range(0,prob.m+1)]
avg = [(prob.L[i]+prob.U[i])/2.0 for i in range(0,prob.m+1)]
nexp.initMRAS(avg, std, 500, 10, 0.0)
nexp.search(21, 0)
gamma = nexp.gamma(nexp.best)
rron = [nexp.Ron(s[0], s[1], nexp.best) for s in nexp.scen]
roff = [s[2] for s in nexp.scen]
ns = len(nexp.scen)
grof = [roff[i]*gamma for i in range(ns)]
plot(sump, roff, '-.', label='Offline Revenue')
plot(sump, rron, label='Online Revenue')
plot(sump, grof, ':', label='Guaranteed Revenue')

std = [prob.U[i] for i in range(0,prob.m+1)]
avg = [(prob.L[i]+prob.U[i])/2.0 for i in range(0,prob.m+1)]
nexp.initMRAS(avg, std, 200, 10, 0.4 )
nexp.search(21, 1)
regret = nexp.regret(nexp.best)
rron = [nexp.Ron(s[0], s[1], nexp.best) for s in nexp.scen]
grof = [roff[i]-regret for i in range(ns)]
#plot(sump, rron, '-.', label='AR online')
#plot(sump, grof, ':', label='AR guarantee')

#std = [prob.U[i] for i in range(0,prob.m+1)]
#avg = [(prob.L[i]+prob.U[i])/2.0 for i in range(0,prob.m+1)]
#nexp.initMRAS(avg, std, 200, 10, 0.4 )
#nexp.search(21, 2)
#regret = nexp.regret(nexp.best)
#rron = [nexp.Ron(s[0], s[1], nexp.best) for s in nexp.scen]
#grof = [roff[i]-regret for i in range(ns)]
#plot(rron, label='WAR online')
#plot(grof, ':', label='WAR guarant')

xlim(ltot, utot)
legend(loc=0)
xlabel("Total number of requests in extreme profiles")
ylabel("Revenue ($)")
savefig("dist-cr-mras.eps")

figure()
pent = [s[3][0] for s in nexp.scen]
penf = [nexp.ff.g(pent[i]) for i in range(ns)]
pend = [nexp.ff.g(pent[i]+1)-penf[i] for i in range(ns)]
plot(pent, 'o')
for j in range(1, nexp.m):
    pent = [pent[i] - s[3][j] for i,s 
					in enumerate(nexp.scen)]
    plot(pent)

xlabel("Offline # accepted for Extreme Inputs")

figure()
plot(penf, 'ro:')
plot(pend)
xlabel("Penalties for Extreme Inputs")

show()
