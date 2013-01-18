from msom_ex2_conf import *

xlab = 'mean no-show rate'

#glidePrint(sina,sinb,0,7,6.0)
#glideSim(sina, sinb, 0, 7, 6.0, 9000, xlab,'msom-ex2-pshift.pkl', mymeth)

custom, vvv, relat, waste, bumps = loadResults('msom-ex2-pshift.pkl')
nm = len(custom)
DISP = 'msom-ex2-pshift'
#DISP = None 
params = {'axes.labelsize': 10,
             'text.fontsize': 10,
             'legend.fontsize': 10,
             'xtick.labelsize': 8,
             'ytick.labelsize': 8}
rcParams.update(params)
figure(figsize=(6,4))
for i in range(nm):
    plot(vvv, relat[i], style[i], label=custom[i])

legend(loc=0, numpoints = 2, markerscale = 0.9)
xlabel(xlab)
if bAbsolute: ylabel('Average net revenues')
else: ylabel('Relative Performance to EMSR/CR (%)')
ylim(6500, 6750) #Cut off some of EMSR/NO

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
else: show()
