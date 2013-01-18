from noshow import *
config.LBH=True
config.BINO=True
config.svl =0.001
#mymeth = ('OBSA/CR','EMSR/CR','EMSR/NV', 'CRSA/NV', 'DP/LBH', 'EMSR/SL')
mymeth = ('EMSR/HCR', 'EMSR/HAR', 'EMSR/NV', 'DP/LBH', 
		   'EMSR/SL', 'EMSR/NO', 'HCR/OSA', 'HAR/OSA')

sina = SimScena()
sina.C = 100.0
sina.m = 2
sina.beta = 0.2
sina.p = (0.0, 0.0)
sina.f = (100.,40.)
sina.V = 900.0
#vin = 5.0
#sina.f[0]*(1+sina.p[1]*sina.beta/(1-sina.p[1]))*vin
sina.L = (40.,70.)
sina.U = (80.,130.)
print "Demand Factor: ", sina.demandFactor(), "V=", sina.V

sinb = copy(sina)
sinb.p = (0.2, 0.2)
print "Demand Factor: ", sinb.demandFactor()

sina.nid = 0.0
sinb.nid = 0.2
xlab = 'no-show rate'

#glidePrint(sina,sinb,0,7,6.0)

glideSim(sina, sinb, 0, 6, 5.0, 10000, xlab,'dist-ex2-pshift.pkl', mymeth)

custom, vvv, relat, waste, bumps = loadResults('dist-ex2-pshift.pkl')
nm = len(custom)
DISP = None #'dist-ex2-pshift'
params = {'axes.labelsize': 10,
             'text.fontsize': 10,
             'legend.fontsize': 10,
             'xtick.labelsize': 8,
             'ytick.labelsize': 8}
rcParams.update(params)

figure(figsize=(6,4))
#ylim(5500, 6500) #Cut off some of EMSR/NO
for i in range(nm):
    plot(vvv, relat[i], style[i], label=custom[i])

legend(loc=0, numpoints = 2, markerscale = 0.9)
xlabel(xlab)
if bAbsolute: ylabel('Average net revenues')
else: ylabel('Relative Performance to EMSR/CR (%)')
if DISP!=None: savefig(DISP+'1.eps')

figure(figsize=(6,4))
for i in range(nm):
    plot(vvv, waste[i], style[i], label=custom[i])
legend(loc=0, numpoints = 2, markerscale = 0.9)
xlabel(xlab)
ylabel('Average unused inventory (per 100)')
#title('Average Empty Seats per 100')
if DISP!=None: savefig(DISP+'2.eps')

figure(figsize=(6,4))
for i in range(nm):
    plot(vvv, bumps[i], style[i], label=custom[i])
legend(loc=0, numpoints = 2, markerscale = 0.9)
xlabel(xlab)
ylabel('Average service denials (per 10,000)')
#title('Average Bumpings per 10,000')
if DISP!=None: savefig(DISP+'3.eps')

if DISP is None: show()
