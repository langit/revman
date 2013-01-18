from noshow import *
p = (0.10, 0.20) #no show rate
B = 120
S = 70
fa = newvc.UnifBinoMass(1.-p[1], 1.-p[0], B, B)
xx = [i for i in range(S, B+1)]
ma = [fa.choose(i) for i in xx]
fb = newvc.BinomialMass(1.-sum(p)/2.0, B, B)
xc = newvc.getquant(fb, 0, B, 0.05)
mb = [fb.choose(i) for i in xx]
print xc, sum(mb[xc[0]-S:xc[1]-S+1])
mc = [1./(xc[1]-xc[0]+1) for i in range(len(xc))]
xc.insert(0, xc[0])
xc.append(xc[len(xc)-1])
mc.insert(0, 0.0)
mc.append(0.0)
plot(xx, ma, 'ro-', label='Actual PMF')
plot(xx, mb, 'bs:', label='Input to NV')
plot(xc, mc, 'gd:', label='Bounds to CR')
#xlabel("No-show: [%(p0).2f, %(p1).2f], Total Bookings: %(B)d"%{"p0":p[0],"p1":p[1],"B":B})
xlabel("number of customers showing up out of a total of 120")
ylabel("probability mass")
legend(loc=0)
#show()
savefig('pmfcmp.eps')
