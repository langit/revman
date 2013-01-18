from random import * 
from pylab import *
from copy import *

import newvc

class NestedPolicy:
    def __init__(me, x, prob):
        me.C = int(prob.C)
        me.f = prob.f
        me.V = prob.V
        me.beta = prob.beta
        me.x = x
        m = len(x) -1
        me.b = x[:]
        for i in range(m-1, 0, -1):
            me.b[i] += me.b[i+1]

    def getVC(me):
        return me.b[1]

    def freeze(me, nest=True):
        b = me.b[:] #make a copy
        i = len(b) - 1
        while i > 0:
            bfloor= int(b[i])
            rand = bfloor+ random()
            j = i 
            while j>0  and b[j] < rand:
                b[j] = bfloor
                j -= 1
            bceil = bfloor + 1
            while j>0 and b[j] < bceil:
                b[j] = bceil
                j -= 1
            i = j
        if nest: return b
        for i in range(1, len(b)-1):
            b[i] = b[i] - b[i+1]
        return b

    def testFreeze(me, rep):
        if rep < 1: return
        h = me.freeze()
        rep = int(rep)
        for i in range(rep):
            x = me.freeze()
            for j in range(len(h)):
                h[j] += x[j]
        rep = float(rep+1)
        for j in range(len(h)):
            h[j] = float(h[j])/rep
        print "avg: ", h
        print me.b

    def execPolicy(me, seq, p):
        x = me.freeze(False)
        rv = 0.0
        ax = 0
        for i in seq:
            for j in range(i,  len(x)):
                if x[j] > 0:
                    x[j] -= 1
                    rv += me.f[i]
                    ax += 1
                    break
        rv *= (1 - p + me.beta*p)
        ax = (1-p) * ax - me.C
        if ax > 0: #bumped
            rv -= ax * me.V
        return rv,ax

    def execBino(me, seq, p):
        """seq has noshows marked as negative."""
        x  = me.freeze(False)
        rv = 0.0
        ax =  - me.C
        for i in seq:
            nosh = (i < 0) #no show
            if nosh: i = -i
            for j in range(i,  len(x)):
                if x[j] > 0: #accept
                    x[j] -= 1
                    if nosh:
                        rv += me.f[i] * me.beta
                    else:
                        rv += me.f[i]
                        ax += 1
                    break
        if ax > 0: #bumped
            rv -= ax * me.V
        return rv,ax

class FCFSPolicy:
    def __init__(me, vc, prob):
        me.C = int(prob.C)
        me.f = prob.f
        me.V = prob.V
        me.beta = prob.beta
        me.vc = int(vc) #virtual capacity

    def execPolicy(me, seq, p):
        ax = 0
        rv = 0.0
        for i in seq:
            if ax >= me.vc: break
            ax += 1
            rv += me.f[i]
        rv *= (1 - p + me.beta*p)
        ax = (1-p) * ax - me.C
        if ax > 0: #bumped
            rv -= ax * me.V
        return rv,ax

    def execBino(me, seq, p):
        ax = - me.C
        quota = me.vc
        rv = 0.0
        #print "fares:", me.f
        for i in seq:
            if quota <= 0: break
            quota -= 1
            nosh = (i < 0) #no show
            if nosh:
                rv += me.f[-i] * me.beta
            else:
                rv += me.f[i]
                ax += 1
            #print quota, ax, i, rv

        if ax > 0: #bumped
            rv -= ax * me.V
        return rv,ax

class OfflinePick:
    def __init__(me, x, f):
        me.x = x
        me.f = f
        me.i = 1
        me.left = x[1]

    def pick(me):
        while me.left < 1:
            me.i += 1
            if me.i >= len(me.x):
                return 0.0
            me.left = me.x[me.i]
        me.left -= 1
        return me.f[me.i]

class OfflineOpt:
    def __init__(me, prob):
        me.C = int(prob.C)
        me.f = prob.f
        me.V = prob.V
        me.beta = prob.beta
        me.m = prob.m

    def execPolicy(me, seq, p):
        prof = [0 for i in range(me.m + 1)]
        for i in seq: prof[i] += 1
        vc = int(me.C / (1-p))
        rv = 0.0
        for i in range(1,len(me.f)):
            use = prof[i]
            if use > vc:
                use = vc
            rv += use * me.f[i]
            vc -= use
        return rv*(1-p+me.beta*p), -vc*(1-p)

    def execBino(me, seq, p):
        x = [0 for i in range(me.m+1)]
        for i in seq:
            if i > 0: x[i] += 1
            else: x[-i] += 1
        vc = me.newsVCL(x, 1.0 - p)
        for i in range(1,len(x)):
            use = x[i]
            if use > vc:
                use = vc
            x[i] = use
            vc -= use

        #below is copied from NestedPolicy.execBino
        rv = 0.0
        ax =  - me.C
        for i in seq:
            nosh = (i < 0) #no show
            if nosh: i = -i
            for j in range(i,  len(x)):
                if x[j] > 0: #accept
                    x[j] -= 1
                    if nosh:
                        rv += me.f[i] * me.beta
                    else:
                        rv += me.f[i]
                        ax += 1
                    break
        if ax > 0: #bumped
            rv -= ax * me.V
        return rv,ax

    def newsVCL(me, x, rho=None, vbs = False): #Lan's Version
        """Given profile, decide on VC!"""
        
        C = b = int(me.C)
        B = int(sum(x))
        if B<=C: return C
        if rho == None: rho = 1. - sum(me.p)/2.0
        fact = 1.0 + me.beta*(1-rho)/rho
        if vbs: print 'C=%i, B=%i, rho=%f'%(C,B,rho)
        fs = newvc.BinomialMass(rho, C, B) #show-up distr.
        prob = fs.choose(C)
        pp = OfflinePick(x, me.f)
        for i in range(C): pp.pick()
        test = fact * pp.pick() / me.V
        if vbs: print 'VC: b',  "\tf'_i/V",'\tP{C-}rho', '\tP{s|b>=C}'
        while prob < test:
            T = fs.choose(C-1)*rho
            if vbs: print '%d\t%f\t%f\t%f'%(b,test,T,prob)
            prob += T
            b += 1
            fs.expand()
            test = fact * pp.pick() / me.V
            if b == B: break
        if vbs:
            print 'P{s|b>=C}>=f/V: %f >= %f'%(prob, test)
            if b == B: print '*May not hold since b = B = %i.' % b
        return b


class UniformSource:
    def __init__(me, U, L):
        me.U = U
        me.L = L
        me.m = len(U)

    def triangle(me, i, Left):
        t = math.sqrt(uniform(0,1))
        if Left: t = 1.0 - t
        return me.L[i] + (me.U[i] +1. - me.L[i])*t

    def nextSeq(me, LBH = True, COR=True):
        if COR:
            Left = (uniform(0,1) > 0.5)
            prof = [int(me.triangle(i, Left)) for i in range(me.m)]            
        else:
            prof = [int(uniform(me.L[i],me.U[i]+1)) for i in range(me.m)]
        
        sequ = [0 for i in range(sum(prof)) ]
        psum = prof[0]
        curr = 1
        maxi = len(sequ) -1
        for i in range(len(sequ)):
            if i >= psum:
                psum += prof[curr]
                curr += 1
            sequ[maxi - i] = curr
        if LBH: return sequ
        return sample(sequ,len(sequ))


class LimitInfo:
    def __init__(me, C, f, L, U, V, p, beta):
        me.C = C #capacity
        me.m = len(f) # # of fares
        me.f = (V*(1-p[0]),)+f+(0,) #fares
        me.U = (0,)+U
        me.L = (0,)+L
        me.V = V
        me.beta = beta
        me.p = p
        me.h =[1.0-me.p[i]*(1.0 - me.beta) for i in range(2)]

    def offopt(me, castID, canID):
        vc = me.C/(1.0-me.p[canID]) # virtual capacity
        used = rev = qast = 0.0
        for i in range(1,me.m+1):
            if i < castID:
                qast = me.L[i]
            else:
                qast = me.U[i]
            used += qast
            if used > vc: used = vc
            rev += (me.f[i] - me.f[i+1]) * used
        return rev*me.h[canID]

    def nleft(me, castID):
        vc = me.C/(1-me.p[0])
        for i in range(1,castID):
            vc -= me.L[i]
        #if vc < 0: vc = 0
        return vc

    def rplus(me, castID):
        #if castID < 2: return 0
        rev = 0
        for i in range(1,castID):
            rev += me.f[i] * me.L[i]
        return rev*me.h[1]


    def compg(me, castID):
        if castID<1:
            diff = me.h[1]*me.Roff[0]-me.h[0]*me.Roff[1]
        else:
            diff = me.Roff[castID] - me.Roff[castID+1]
        return diff/(me.h[1]*me.f[castID])

    def prepare(me):
        me.Roff = [me.offopt( i or 1, i and 1) for i in range(me.m+1)] #R^*
        me.N = [me.nleft(i) for i in range(me.m+1)] #N_j
        me.Radd = [me.rplus(i) for i in range(me.m+1)] #R^+
        me.g = [me.compg(i) for i in range(me.m)]
        me.sumg = [0 for i in range(me.m+1)]
        for i in range(1,me.m+1):
            me.sumg[i] = me.sumg[i-1]+me.g[i-1]

    def findu(me): #make sure me.prepare() was called
        u,v = 1,2
        while me.Radd[v]*me.sumg[v] < me.Roff[v]*me.N[v]:
            u,v = v,v+1
            if u == me.m: break
        return u

    def gamma(me, u): #make sure me.prepare() was called
        gamma = me.Radd[u]/(me.h[1]*me.f[u]) + me.N[u]
        gamma /= me.Roff[u]/(me.h[1]*me.f[u]) + me.sumg[u]
        return gamma

    def policyCR(me, u, gamma):
        x = [0 for i in range(me.m+1)]
        x[0] = - gamma*me.g[0]*(1.-me.p[0])
        for i in range(1,u):
            x[i] = me.L[i]+gamma*me.g[i]
        x[u] = me.Roff[u] * gamma - me.Radd[u]
        x[u]/= me.h[1]*me.f[u]
        return x

    def findRev(me, x, cast, t):
        rev = 0.0
        acc = 0.0
        lmt = 0.0
        for k in range(me.m, 0, -1):
            if k >= cast: nac = me.U[k] + acc
            else: nac = me.L[k] + acc
            lmt += x[k]
            if nac > lmt: nac = lmt
            rev += (nac-acc)*me.f[k]
            acc  = nac
        rev *= me.h[t]
        acc *= (1. - me.p[t])
        if acc > me.C:
            rev -= me.V * (acc - me.C)
        return rev

    def findCR(me, x):
        gamma = 1.0
        for cast in range(1, me.m+1):
            gan = me.findRev(x, cast, 1)/me.Roff[cast]
            if gan < gamma: gamma = gan
        gan = me.findRev(x, 1, 0)/me.Roff[0]
        if gan < gamma: gamma = gan
        return gamma

    #below we compute EMSR policy with OB
    def invUniFbar(me, p, i):
        return me.L[i] * p + me.U[i] * (1-p)
    
    def policyEMSR(me, vc): #virture capacity
        x = [0 for i in range(me.m+1)]
        for j in range(1, me.m+1):
            x[j] = vc
            for i in range(1, j):
                sij  = me.invUniFbar(me.f[j]/me.f[i], i)
                x[j] -= sij
            if x[j] <= 0.0:
                x[j] = 0.0
                break
        for i in range(1, me.m):
            x[i] = x[i]-x[i+1]
        return x


class SimScena:
    def __init__(me):
        pass

    def getLoadFactor(me):
        return (sum(me.U)+sum(me.L))/2.0/me.C

    def makeProblem(me):
        return LimitInfo(me.C, me.f, me.L, me.U, me.V, me.p, me.beta)

    def makeSource(me):
        return UniformSource(me.L, me.U)

    #below we find Newsvendor Virtual Capacity
    def faverage(me):
        u = [(me.L[i]+me.U[i])/2.0 for i in range(len(me.L))]
        ff = 0.0
        for i in range(len(u)):
            ff += u[i] * me.f[i]
        return ff/sum(u)

    def newsVC(me, rho=None, vbs = False):
        C = b = int(me.C)
        B = int( me.C/(1.0-me.p[1]) )+1
        if rho == None: rho = 1. - sum(me.p)/2.0
        tes = 1.0 + me.beta*(1-rho)/rho
        tes = tes*me.faverage()/me.V
        if vbs: print 'C=%i, B=%i, rho=%f, tes=%f'%(C,B,rho,tes)
        Fd = newvc.totalCDF(me.L, me.U) #demand distr.
        fs = newvc.BinomialMass(rho, b, B) #show-up distr.
        prob = 1. - Fd.mass(C-1)
        if prob < tes: return B
        prob *= fs.choose(C)
        if vbs: print 'VC: b', '\t1-F(b)', '\trho*P{}', '\tP{s|b>=C}'
        while prob < tes:
            S = 1. - Fd.mass(b)
            T = fs.choose(C-1)*rho
            if vbs: print '%d\t%f\t%f\t%f'%(b, S,T,prob)
            prob += S*T
            b += 1
            fs.expand()
            if b == B: break
        if vbs:
            print 'P{s|b>=C}>=f/V: %f >= %f'%(prob, tes)
            if b == B: print '*May not hold since b = B = %i.' % b
        return b

    def newsVCL(me, rho=None, vbs = False): #Lan's Version
        C = b = int(me.C)
        B = int( me.C/(1.0-me.p[1]) )+1
        if rho == None: rho = 1. - sum(me.p)/2.0
        tes = 1.0 + me.beta*(1-rho)/rho
        tes = tes*me.faverage()/me.V
        if vbs: print 'C=%i, B=%i, rho=%f, tes=%f'%(C,B,rho,tes)
        fs = newvc.BinomialMass(rho, C, B) #show-up distr.
        prob = fs.choose(C)
        if vbs: print 'VC: b',  '\tP{C-}rho', '\tP{s|b>=C}'
        while prob < tes:
            T = fs.choose(C-1)*rho
            if vbs: print '%d\t%f\t%f'%(b,T,prob)
            prob += T
            b += 1
            fs.expand()
            if b == B: break
        if vbs:
            print 'P{s|b>=C}>=f/V: %f >= %f'%(prob, tes)
            if b == B: print '*May not hold since b = B = %i.' % b
        return b


class ScenaGlide:
    def __init__(me, a, b):
        if a.m != b.m:
            raise "diff. number of fares!"
        me.a = a
        me.b = b
        
    def getGlide(me, fract):
        c = SimScena()
        af = 1.0 - fract
        bf = fract
        c.nid = me.a.nid*af+me.b.nid*bf
        c.C = me.a.C * af + me.b.C * bf
        c.m = me.a.m
        c.beta = me.a.beta * af + me.b.beta * bf
        c.V = me.a.V * af + me.b.V * bf
        c.f = tuple([me.a.f[i]*af+me.b.f[i]*bf for i in range(c.m)])
        c.L = tuple([me.a.L[i]*af+me.b.L[i]*bf for i in range(c.m)])
        c.U = tuple([me.a.U[i]*af+me.b.U[i]*bf for i in range(c.m)])
        c.p = (me.a.p[0]*af+me.b.p[0]*bf, me.a.p[1]*af+me.b.p[1]*bf)
        return c

class PolicyBed:
    def __init__(me, policy, BINO):
        me.plc = policy
        me.BINO = BINO
        me.res = [0.,0.,0.]

    def reset(me):
        for i in range(len(me.res)):
            me.res[i] = 0.

    def execute(me, seq, rp):
        if me.BINO:
            bio = me.plc.execBino(seq, rp)
        else:
            bio = me.plc.execPolicy(seq, rp)
        me.res[0] += bio[0]
        if bio[1] < 0:
            me.res[1] -= bio[1] #empty seats
        else:
            me.res[2] += bio[1] #bumbed

    def getResult(me, reps, off):
        for i in range(3):
            me.res[i] /= off
            off = reps
        return me.res

meth = ['OffOpt','CR/OBSA', 'EMSR/CR', 'EMSR/NV', 'CRSA/NV', 'FCFS/NV']
style = ['yx-', 'rs-', 'gs-', 'go:', 'k+--', 'bx-.']

def binoSeq(seq, rp):
    nosh = 0
    for i in range(len(seq)):
        if random()<rp:
            seq[i] = -seq[i]
            nosh += 1
    return nosh

#### MODULE CONFIG ####
LBH=False
COR=False
ptype=2
BINO=True
#### MODULE CONFIG ####

def makeCROB(prob):
    prob.prepare()
    u = prob.findu()
    gamma = prob.gamma(u)
    x  = prob.policyCR(u,gamma)
    vc = sum(x[1:])
    print "ratio:", gamma, "Allocated Seats: ", vc
    print "x=", x
    return NestedPolicy(x, prob)

def makeEMSR(prob, vc):
    x = prob.policyEMSR(vc)
    print "EMSR:  Allocated Seats: ", sum(x[1:])
    print "x=", x, "True Gamma: ", prob.findCR(x)
    return NestedPolicy(x, prob)

def makeCRVC(prop, vc):
    prop.p = (0.0, 0.0)
    prop.C, vc = vc, prop.C
    prop.prepare()
    u = prop.findu()
    gamma = prop.gamma(u)
    x  = prop.policyCR(u,gamma)
    print "ratio:", gamma, "Allocated Seats: ", sum(x[1:])
    print "x=", x, "True Gamma: ", prob.findCR(x)
    prop.C = vc
    return NestedPolicy(x, prop)


def scenaSim(scen, reps):
    print "||========>> Start New Scenario..."
    prob = scen.makeProblem()
    np = makeCROB(prob)
    vc = np.getVC()
    np = PolicyBed(np, BINO)

    tp = makeEMSR(prob, vc)
    tp = PolicyBed(tp, BINO)

    pm = sum(scen.p)/2.
    if ptype == 0:
        pm = pm * 2 / 3
    elif ptype == 1:
        pm = pm * 4 / 3

    vc = scen.newsVCL(1.-pm, True)
    ep = prob.policyEMSR(prob, vc)
    ep = PolicyBed(ep, BINO)

    prop = scen.makeProblem()
    pp = makeCRVC(prop)
    pp = PolicyBed(pp, BINO)
    
    ff = FCFSPolicy(vc, prob)
    ff = PolicyBed(ff, BINO)

    op = OfflineOpt( prob )

    src = scen.makeSource()

    print 'give a test of op.newsVCL()'
    seq = src.nextSeq(LBH, COR)
    x = [0 for i in range(len(prob.f))]
    for i in seq: x[i] += 1
    op.newsVCL(x, 1.0 - pm, True)
    nosh = binoSeq(seq, pm)
    print pm, nosh, len(seq) 

    op = PolicyBed(op, BINO)    
    pls = [op, np, tp, ep, pp, ff]
    for i in range(reps):
        seq = src.nextSeq(LBH, COR)
        rp = uniform(0., 1.)
        if ptype < 2:  rp = math.sqrt(rp) #mode at 1
        if ptype < 1:  rp = 1. - rp # mode at 0
        rp = scen.p[0] * (1. - rp) +scen.p[1]*rp
        if BINO: binoSeq(seq, rp)

        for k in range(len(pls)):
            pls[k].execute(seq, rp)

    off = pls[0].res[0]
    for k in range(len(pls)):
        pls[k] = pls[k].getResult(reps, off)
        print meth[k], pls[k]
    return pls

def barChart(data, label):
    locats = [i for i in range(1, len(data)+1)]
    bar(locats, data)
    xticks([x + 0.4 for x in locats], label) 

#barChart([2, 3, 1], ['Tom', 'Barbara', 'Mark'])

def glideSim(sa, sb, fa, fb, grids, reps, xlab, DISP=True):
    glide = ScenaGlide(sa,sb)
    nm = len(meth)
    relat = [[] for i in range(nm)]
    waste = [[] for i in range(nm)]
    bumps = [[] for i in range(nm)]
    vvv = [k/grids for k in range(fa, fb)]
    for k in range(len(vvv)):
        sinf = glide.getGlide(vvv[k])
        vvv[k] = sinf.nid
        resu = scenaSim(sinf, reps)
        for i in range(nm):
            relat[i].append(resu[i][0])
            waste[i].append(resu[i][1])
            bumps[i].append(resu[i][2])
    #print relat
    for i in range(nm):
        plot(vvv, relat[i], style[i], label=meth[i])
    #axis([1., 2., .8, .94])
    legend(loc=3)
    xlabel(xlab)
    ylabel('Competitve Ratio')
    title('Competitve Ratio')
    
    figure()
    for i in range(nm):
        plot(vvv, waste[i], style[i], label=meth[i])
    legend()
    xlabel(xlab)
    ylabel('Average Empty Seats')
    title('Average Empty Seats')
    
    figure()
    for i in range(nm):
        plot(vvv, bumps[i], style[i], label=meth[i])
    legend()
    xlabel(xlab)
    ylabel('Average Bumpings')
    title('Average Bumpings')
    if DISP: show()


###TODO: show a picture of the booking limits!
###TODO: the average limits of optimal policies -- as a policy!
