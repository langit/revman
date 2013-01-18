from noshow import *
p = (0.1, 0.3) #no show rate
B = 100
S = 60
params = {'axes.labelsize': 10,
        'text.fontsize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8}
rcParams.update(params)

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
plot(xx, mb, 'bs:', label='Binomial')
plot(xc, mc, 'gd:', label='Quantiles (5%, 95%)')
#xlabel("No-show: [%(p0).2f, %(p1).2f], Total Bookings: %(B)d"%{"p0":p[0],"p1":p[1],"B":B})
xlabel("Number of show-ups out of %(B)d reservations in total"%{"B":B})
ylabel("probability mass")
legend(loc=0)
#show()
savefig('msom-ex7-pmf.eps')
