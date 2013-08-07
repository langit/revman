import math
from pylab import *
from copy import *
from random import * 
import newvc
import pickle

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

class OffOptPolicy:
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
        for i in range(1,len(prof)):
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

    def newsVCL(me, x, rho=None, vbs = 0): #Lan's Version
        """Given profile, decide on VC!"""
        
        C = b = int(me.C)
        B = int(sum(x))
        if B<=C: return C
        if rho == None: rho = 1. - sum(me.p)/2.0 #assume: E(p)=p0/2+p1/2
        fact = 1.0 + me.beta*(1-rho)/rho
        if vbs>0: print 'C=%i, B=%i, rho=%f'%(C,B,rho)
        fs = newvc.BinomialMass(rho, C, B) #show-up distr.
        prob = fs.choose(C)
        pp = OfflinePick(x, me.f)
        for i in range(C): pp.pick()
        test = fact * pp.pick() / me.V
        if vbs>1: print 'VC: b',  "\tf'_i/V",'\tP{C-}rho', '\tP{s|b>=C}'
        while prob < test:
            T = fs.choose(C-1)*rho
            if vbs>1: print '%d\t%f\t%f\t%f'%(b,test,T,prob)
            prob += T
            b += 1
            fs.expand()
            test = fact * pp.pick() / me.V
            if b == B: break
        if vbs>0:
            print 'P{s|b>=C}>=f/V: %f >= %f'%(prob, test)
            if b == B: print '*May not hold since b = B = %i.' % b
        return b

class UniformSource:
    def __init__(me, scen, ddr, pdr):
        me.m = len(scen.U)
        me.U = [scen.U[i]/ddr for i in range(me.m)]
        me.L = [scen.L[i]/ddr for i in range(me.m)]
        me.p = [scen.p[i]/pdr for i in range(2)]

    def triangle(me, i, Left):
        """the CDF is x**2, x in [0,1] -- right biased"""
        t = math.sqrt(uniform(0,1)) #right biased
        if Left: t = 1.0 - t #left biased
        return me.L[i] + (me.U[i] +1. - me.L[i])*t

    def nextProfile(me, cor):
        if cor: #demand for different fares are correlated 
            #0.5: half left triangle, half right triangle
            #when mix the two equally, you get uniform!
            #so each individual demand is still uniform.
            Left = (uniform(0,1) > 0.5) #tend toward lower bounds
            prof = [int(me.triangle(i, Left)) 
                            for i in range(me.m)]
        else:
            prof = [int(uniform(me.L[i],me.U[i]+1)) 
                            for i in range(me.m)]
        return prof

    def nextSeq(me, lbh=True, cor=True):
        prof = me.nextProfile(cor)
        sequ = [0 for i in range(sum(prof)) ]
        psum = prof[0]
        curr = 1
        maxi = len(sequ) -1
        for i in range(len(sequ)):
            if i >= psum:
                psum += prof[curr]
                curr += 1
            sequ[maxi - i] = curr
        if lbh: return sequ
        return sample(sequ,len(sequ))

    def nextRate(me, typ):
        if me.p[0]==me.p[1]: return me.p[0]
        rp = uniform(0., 1.) #mode: uniform
        if typ < 2:  rp = math.sqrt(rp) 
        if typ < 1:  rp = 1. - rp 
        return me.p[0] * (1. - rp) + me.p[1]*rp

class LimitInfo:
    def __init__(me, C, f, L, U, V, p, beta):
        me.C = C #capacity
        me.m = len(f) # # of fares
        me.f = (V*(1-p[0]),)+f+(0,) #fares
        me.U = (0,)+U
        me.Ut= sum(U)
        me.L = (0,)+L
        me.V = V
        me.beta = beta
        me.p = p
        me.h =[1.0-p[i]*(1.0 - beta) for i in range(2)]
        me.prepared = False

    def binoAdj(me):
        pav = sum(me.p)/2.0
        std = 3.*sqrt(pav*(1.-pav)/me.C)
        p0  = me.p[0] - std * me.p[0]
        p1  = me.p[1] + std * (1.-me.p[1])
        adj = copy(me)
        adj.p = (p0,p1)
        return adj

    def binomAdj(me):
        """adjust interval p when we set up binomial dist:
when config.BINO is True, the resultant conf. interv. of NSR 
is usually larger than the NSR interval from which a rate
is uniformly sampled as the failure probability of a Binomial
distribution. This is for the model in the MSOM paper."""
        rho = sum(me.p)/2.0
        C = int(me.C)
        fs = newvc.UnifBinoMass(me.p[0], me.p[1], C, C)
        if fs.delta < 1e-8: fs = newvc.BinomialMass(rho, C, C)
        p = newvc.getquant(fs, 0, int(C), config.confid)
        print "binomAdj:", me.p,
        me.p = (p[0]/float(C), p[1]/float(C))
        print me.p

    def offopt(me, castID, canID):
        print me.p, castID, canID,
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
        print me.h[canID], rev
        return rev*me.h[canID]

    def nleft(me, castID):
        vc = me.vc
        for i in range(1,castID):
            vc -= me.L[i]
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
        if me.prepared: return
        me.vc = min(sum(me.U)-me.U[0], me.C/(1.-me.p[0]))
        me.Roff = [me.offopt( i or 1, i and 1) for i in range(me.m+1)] #R^*
        me.N = [me.nleft(i) for i in range(me.m+1)] #N_j
        me.Radd = [me.rplus(i) for i in range(me.m+1)] #R^+
        me.g = [me.compg(i) for i in range(me.m)]
        me.sumg = [0 for i in range(me.m+1)]
        for i in range(1,me.m+1):
            me.sumg[i] = me.sumg[i-1]+me.g[i-1]
        me.prepared = True

    def dump(me):
        print "L =", me.L
        print "U =", me.U
        print "p =", me.p
        print "R*=", me.Roff
        print "R+=", me.Radd
        print "Nj=", me.N
        print "gj=", me.g
        print "sumg:", me.sumg

    def findu(me): #make sure me.prepare() was called
        u,v = 1,2
        while me.Radd[v]*me.sumg[v] < me.Roff[v]*me.N[v]:
            u,v = v,v+1
            if u == me.m: break
        return u

    def gamma(me, u): #make sure me.prepare() was called
        gamma = me.Radd[u]/(me.h[1]*me.f[u]) + me.N[u]
        gamma /= me.Roff[u]/(me.h[1]*me.f[u]) + me.sumg[u]
        if gamma > 1.0000001: 
            me.dump(); print u, gamma
            raise "gamma greater than 1: "+str(gamma)
        return gamma

    def policyCR(me, u, gamma):
        x = [0 for i in range(me.m+1)]
        x[0] = - gamma*me.g[0]*(1.-me.p[0])
        for i in range(1,u):
            x[i] = me.L[i]+gamma*me.g[i]
        x[u] = me.Roff[u] * gamma - me.Radd[u]
        x[u]/= me.h[1]*me.f[u]
        return x

    #the following three methods are for MAR
    def findut(me): #make sure me.prepare() was called
        u,v = 1,2
        while me.sumg[v] < me.N[v]:
            u,v = v,v+1
            if u == me.m: break
        return u

    def regret(me, u): #make sure me.prepare() was called
        regret = (me.N[u] - me.sumg[u]) * me.f[u] * me.h[1]
        regret = me.Roff[u] - me.Radd[u] - regret
        return regret

    def policyAR(me, u):
        x = [0 for i in range(me.m+1)]
        x[0] = - me.g[0]*(1.-me.p[0])
        for i in range(1,u):
            x[i] = me.L[i]+me.g[i]
        x[u] = me.N[u] - me.sumg[u]
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
        gamma = 2.0
        for cast in range(1, me.m+1):
            gan = me.findRev(x, cast, 1)/me.Roff[cast]
            if gan < gamma: gamma = gan
        gan = me.findRev(x, 1, 0)/me.Roff[0]
        if gan < gamma: gamma = gan
        return gamma

class SimScena:
    def __init__(me):
        me.ddr = defDDR
        me.pdr = defPDR

    def initSim(me): pass

    def modifyParam(me, p): pass

    def demandFactor(me):
        #return (1-pmean(me.p))*(sum(me.U)+sum(me.L))/2.0/me.C
        return (sum(me.U)+sum(me.L))/2.0/me.C

    def makeProblem(me):
        p = LimitInfo(me.C, me.f, me.L, me.U, me.V, me.p, me.beta)
        if config.BINO: p.binomAdj()
        return p

    def extendScen(me):
        ext = copy(me)
        ext.m = me.m+1
        ext.f = me.f + (me.f[me.m-1]/10000.0,)
        ext.U = me.U + (me.C/(1.-me.p[1]),)
        ext.L = me.L + (0.,)
        return ext

    def makeSource(me):
        return UniformSource(me, me.ddr, me.pdr)

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

    def newsVCL(me, rho=None, vbs = 0): #Lan's Version
        C = b = int(me.C)
        B = int( me.C/(1.0-me.p[1]) )+1
        if rho == None: rho = 1. - sum(me.p)/2.0
        tes = 1.0 + me.beta*(1-rho)/rho
        tes = tes*me.faverage()/me.V
        if vbs>0: print 'C=%i, B=%i, rho=%f, tes=%f'%(C,B,rho,tes)

        #fs = newvc.BinomialMass(rho, C, B) #show-up distr.
        fs = newvc.UnifBinoMass(1.-me.p[1],1.-me.p[0], C, B) #show-up distr.
        if fs.delta < 1e-8: fs = newvc.BinomialMass(rho, C, B)

        prob = fs.choose(C)
        if vbs>1: print 'VC: b',  '\tP{C-}rho', '\tP{s|b>=C}'
        while prob < tes:
            T = fs.choose(C-1)*rho
            if vbs>1: print '%d\t%f\t%f'%(b,T,prob)
            prob += T
            b += 1
            fs.expand()
            if b == B: break
        if vbs>0:
            print 'P{s|b>=C}>=f/V: %f >= %f'%(prob, tes)
            if b == B: print '*May not hold since b = B = %i.' % b
        return b

    def newsVCDU(me, rho=None, vbs = 0): 
        "Discrete Uniform Version--approx"
        if rho == None: rho = 1. - sum(me.p)/2.0
        tes = 1.0 + me.beta*(1-rho)/rho
        tes = tes*me.faverage()/me.V
        if vbs>0: print 'C=%i, rho=%f, tes=%f'%(me.C,rho,tes)
        #Experiment: NewsVendor 2
        #r/V  <= P{ S(b) >= C }
        #where S(b) = b * rho, 
        #      with rho ~ U(1-p1, 1-p0)
        tes = 1-me.p[1] * tes - me.p[0]*(1-tes)
        return int(me.C/tes+0.9999)

    def newsVCU(me): #Continuous Uniform Version
        #q = 1 - p -- show rate
        q1 = 1. - me.p[0]
        q0 = 1. - me.p[1]
        qv = (q0+q1)/2.0 #average q
        qq = q1*q1 - 2*(q1-q0)*(me.beta + qv - me.beta*qv)*me.faverage()/me.V
        if qq < q0*q0: qq = q0*q0 #critical condition
        qq =  math.sqrt(qq)
        return me.C/qq

    #below we compute EMSR policy with OB
    def invSurvival(me, p, i): #inverse Survival
        return me.L[i-1]*p + me.U[i-1]*(1-p)
    
    def policyEMSR(me, vc): #virture capacity
        x = [0 for i in range(me.m+1)]
        f = (0,)+me.f
        for j in range(1, me.m+1):
            x[j] = vc
            for i in range(1, j):
                sij  = me.invSurvival(f[j]/f[i],i)
                x[j] -= sij
            if x[j] <= 0.0:
                x[j] = 0.0
                break
        for i in range(1, me.m):
            x[i] = x[i]-x[i+1]
        return x

    #set virtual capacity by service level type I
    #assuming uniform noshow rate.
    #ref. van Ryzin RM book pg 141.
    def virtucapSL1(me, denp): #virture capacity
        if config.BINO: return me.binoVCbySL1(denp)
        return me.C/(1.0-me.p[0] - denp*(me.p[1]-me.p[0]))

    def binoVCbySL1(me, denp): 
        #see van Ryzin book pg 141 for defn of SL1
        B = int(me.C/(1-me.p[1])) + 1 
        fs = newvc.UnifBinoMass(1.-me.p[1], 1.-me.p[0], 
                               int(me.C), B) 
        if fs.delta < 1e-8: 
           fs = newvc.BinomialMass(1.-me.p[1], int(me.C), B)
        B = int(me.C)
        while True:
            fs.expand()
            s = 0.0
            for i in range(int(me.C)+1, B+2): 
                s += fs.choose(i)
            if s <= denp: B += 1
            else: return B 

    #set virtual capacity by service level type II
    #assuming uniform noshow rate.
    #ref. van Ryzin RM book pg 142.
    def virtucapSL2(me, edp): #virture capacity
        if config.BINO: return me.binoVCbySL2(edp)
        q0, q1 = 1.0 - me.p[1], 1.0 - me.p[0]
        dqq = q1*q1 - q0*q0
        K = q1 + sqrt(edp*dqq)
        A = q1*q1 - edp*dqq
        return me.C*K/A 

    def binoVCbySL2(me, edp):
       svl2 = [0.0]
       def penadj(std, dep, der): 
           svl2[0]=edp-der
           if svl2[0]>0: return svl2[0]
           else: return 0.0
       B = 2+int(me.C/(1.0 - me.p[1]))
       ff = penub(me.p, 1.0, me.C, B, penadj)
       for k in range(int(me.C),B):
           ff.g(k)
           if svl2[0]<0: return k-1
       raise "binoVCbySL2: level two high %f"%edp

    #Below compute the exact static policy for LBH arrivals
    #assuming demand is discrete uniform with given support.
    #noshow rate is also uniform (continous).
    #or noshow is binomial with uniform probability of noshow
    #In both cases, the termianl cost is CONVEX
    def policyStaticDP(me): #ref. van Ryzin RM book pg 157
        #first test for the uppper bound of overbooking.
        V = 1.0 - sum(me.p)/2.0
        f,L,U = (0,)+me.f,(0,)+me.L,(0,)+me.U
        if V*me.V <= f[1]:
            raise "static model violates COR! %f"%(f[1]/V)
        V = None #variables for Bellman equaiton
        B = 2+int(me.C / (1.0 - me.p[1]))
        if config.BINO: #if binomial distribution
            pf = penub(me.p,me.V,me.C,B,penadj(0,0,0,0))
            V = [-pf.g(i) for i in range(B)]
            #print V[int(me.C):]
        else:
            V = [0.0 for i in range(B)]
            V[B-1]=-me.V*((1.0-sum(me.p)/2.0)*(B-1)-me.C)
            #pf = penun(me.p, me.V, me.C,penadj(0,0,0,0))
            for i in range(1+int(me.C/(1.0-me.p[0])), B-1):
                V[i] = (i - i * me.p[0] - me.C)
                V[i] *= - V[i] * me.V
                V[i] /= float(2*i) * (me.p[1] - me.p[0])
                #print V[i] + pf.g(i), #pf for debug

        b = [0 for i in range(me.m+1)] #policy
        b[0] = len(V) - 1
        for j in range(1,me.m+1):
            b[j] = b[j-1]
            K = float(U[j]-L[j]+1)
            ll, uu = int(L[j]), int(U[j])
            while f[j] < V[b[j]-1] - V[b[j]]: 
                b[j] -= 1
                if b[j] < 1: break 
            if b[j] < 1: break
            #update V: V_j(y) = E[max_{u<=D_j} f_j u + V_{j-1}(y+u)]
            #if V'>0: Vj(y)=E[f_j t+V_{j-1}(y+t)|t=min(Dj,[b_j-y]+)]
            #So we have the following cases (exclusively):
            #if y >= b_j: t=0, Vj(y) = V_{j-1}(y)
            #if y+L_j >= b_j: y+t = b_j, Vj(y)=(b_j-y)f_j+V_{j-1}(b_j)
            #if y+U_j >= b_j: Vj(y)=P(Dj>=b_j-y)((b_j-y)f_j+V_{j-1}(b_j))
            # + sum_{L_j<=Dj<b_j-y} [f_j Dj + V_{j-1}(y+Dj)]/(U_j-L_j+1)
            # OR: Vj(y) = Vj(y+1) + f_j P(Dj >= b_j-y) + 
            # [V_{j-1}(y+L_j) - V_{j-1}(b_j)]/(U_j-L_j+1)
            #if y+U_j < b_j: Vj(y) = Vj(y+1) +
            #  [V_{j-1}(y+L_j)-V_{j-1}(y+1+U_j)]/(U_j-L_j+1)
            W = [V[y] for y in range(len(V))] #case y >= b_j
            for y in range(b[j] - 1, -1, -1):
                if y >= b[j] - L[j]: 
                    W[y] = W[y+1] + f[j]
                elif y >= b[j] - U[j]: 
                    W[y] = W[y+1] + f[j] * \
                        (y+ uu - b[j] +1)/K + \
                        (V[y+ll]-V[b[j]])/K
                else:
                    W[y] = W[y+1] + (V[y+ll]-V[y+1+uu])/K
            V = W
        print "StaticDP:", b
        for i in range(1, me.m):
            b[i] = b[i]-b[i+1]
        return b

#this class is used for msom paper ex7: problems got the
#wrong information about noshow: always assume binomial
#of average p, but the actual distr. is conditional 
#binomial with p uniformaly picked from [p0, p1].
class BinoScena(SimScena):
    def __init__(me):
        me.ddr = defDDR
        me.pdr = defPDR
        if not config.BINO:
           print "BinoScena WARNING: set config.BINO to True"

    def initSim(me): #override
        me.realp = me.p
        me.p = (sum(me.p)/2.0, sum(me.p)/2.0)

    def makeSource(me): #override
        src = UniformSource(me, me.ddr, me.pdr)
        src.p = [me.realp[i]/me.pdr for i in range(2)]
        return src

#this class is used for msom paper ex7v:
#problems got the wrong penalty cost: 
#the actual cost varies, but the
#assumed cost is fixed at $500
class VixScena(SimScena):

    def __init__(me, assumedV):
        me.ddr = defDDR
        me.pdr = defPDR
        me.altV = assumedV

    def initSim(me): #override
        print "[=--=] Init Sim: V=" , me.V
        me.V, me.altV = me.altV, me.V

    def modifyParam(me, policy): 
        policy.V = me.altV #true V

#    def bexpect(me, x): #expected bounds (uniform)

class ScenaGlide:
    def __init__(me, a, b):
        if a.m != b.m:
            raise "diff. number of fares!"
        me.a = a
        me.b = b
        
    def getGlide(me, fract):
        c = copy(me.a)
        af = 1.0 - fract
        bf = fract
        c.nid = me.a.nid*af+me.b.nid*bf
        c.ddr  = me.a.ddr*af +me.b.ddr*bf
        c.pdr  = me.a.pdr*af +me.b.pdr*bf
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
            me.res[2] += bio[1] #bumbed/denied
        return bio

    def average(me, reps):
        for k in range(len(me.res)):
            me.res[k] /= reps
        return me.res

    def levels(me): #protection levels
        x = me.plc.x
        return x[1:] if config.buckets \
				else [sum(x[i:]) for i in range(1,len(x))]

#### MODULE CONFIG ####
class noshowConfig:
	#for tadj
    HCR = 0
    HAR = 1
    HWAR = 2

    def __init__(me):
        me.BINO = False
        me.smul = .5 #multiplier to STD(V)
        me.svl  = 0.001
        me.svlp = 50.
        me.tobj = 0 #0: CR, 1: AR, 2: WAR
        me.tadj = 0
        #LBH: Low Before High
        me.LBH=True
        me.confid = 0.05

        me.samples = 600
        me.elites = 40
        me.vsearch= 10
        me.iters = 51
        me.smoother = 0.4
        #for reporting
        me.buckets = True
        me.basemult = 0 #pure CR
        me.percentile = 5 #5th and 95th percentiles
		#uset: default to sqrt(1/12.)/0.5 (uniform distr)
        #me.uset = sqrt(1./3.) #uncertainty set ratio
        me.uset = .5 #25% & 75% quantiles

    def cfgCE(me):
        return me.samples, me.elites, me.smoother

config = noshowConfig()
#Off/Opt:   Offline optimal policy
#CR/OBSA:   Comp. Ratio analysis for OverBooking & Seat Allocation
#EMSR/CR:   Virtual capacity by CR/OBSA, seat allocation by EMSR
#EMSR/NV:   Virtual capacity by NewsVendor model, seat allocation by EMSR
#EMSR/MP:   Virtual capacity by C/(1-E[p]), seat allocation by EMSR
#CRSA/NV:   Virtual capacity by NewsVendor model, seat allocation by GBM-CR
#meth = ['EMSR/CR', 'OBSA/CR', 'OBSA/AR', 'EMSR/NV', 'CRSA/NV', 'Off/Opt']
#meth = ['DP/LBH', 'EMSR/CR', 'OBSA/CR', 'OBSA/AR', 'EMSR/NV', 'EMSR/SL', 'Off/Opt']
#meth = ['DP/LBH', 'EMSR/CR', 'OBSA/CR', 'OBSA/AR', 'EMSR/NV', 'EMSR/SL']

#NOTE: CRE/OSA  >>> HCR/OSA :Hybrid CR / Overbooking & Seat Alloc
meth = ['DP/LBH', 'EMSR/CR', 'OBSA/CR', 'EMSR/NV', 'EMSR/SL',
                 'HCR/OSA', 'HAR/OSA', 'EMSR/HCR', 'EMSR/HAR']
#style = ['mp--','cD-.','ro--','gs-','cv-.','b^-', 'k+:', 'yx-', 'g+-']
style = ['mD--','cx-','rp:','g+-','b*-','kH-', 'ys:', 'k.-', 'g,-']
#COR: Covariance
COR=False

## ptype: type of distribution of p
#2: uniform, 
#1: triangle mode at 1, 
#0: triangle mode at 0.
ptype=2 
defDDR = 1.0 #default DDR (demand deviation ratio)
defPDR = 1.0 #default PDR (noshow rate p dev ratio)
bAbsolute=True #plot absolute revenues
#### MODULE CONFIG ####

def binoSeq(seq, rp):
    nosh = 0
    for i in range(len(seq)):
        if random()<rp:
            seq[i] = -seq[i]
            nosh += 1
    return nosh

def pmean(vp):
    pm = sum(vp)/2.
    if ptype == 0:
        pm = pm * 2 / 3
    elif ptype == 1:
        pm = pm * 4 / 3
    return pm

def printit(x,vc):
    print '%.1f & %.1f &'%(x[2], vc)

def scenaPrint(scen):
    prob = scen.makeProblem()
    prob.prepare()
    u = prob.findu()
    gamma = prob.gamma(u)
    x  = prob.policyCR(u,gamma)
    vc = sum(x[1:])
    printit(x, vc) #OBSA/CR

    x = scen.policyEMSR(vc)
    printit(x, vc) #EMSR/CR

    x = scen.policyStaticDP()
    print 'StaticDP',
    printit(x, sum(x[1:])) #SaticDP

    print "VC by SL1: ", scen.virtucapSL1(config.svl)
    print "VC by SL2: ", scen.virtucapSL2(config.svl)

    u = prob.findut()
    regret = prob.regret(u)
    x  = prob.policyAR(u)
    vc = sum(x[1:])
    printit(x, vc) #OBSA/AR

    pm = pmean(scen.p)
    vc = scen.newsVCL(1.-pm, 0) 
    x = scen.policyEMSR(vc)
    printit(x, vc) #EMSR/NV

    prop = scen.makeProblem()
    prop.p = (0.0, 0.0)
    prop.C = vc
    prop.prepare()
    u = prop.findu()
    gamma = prop.gamma(u)
    x  = prop.policyCR(u,gamma)
    printit(x, sum(x[1:])) #CRSA/NV

    vc = scen.newsVCU() #test VCU
    #vc = int(scen.C/(1.-pm))
    x = scen.policyEMSR(vc)
    printit(x, sum(x[1:])) #EMSR/MP

def getPolicies(scen, custom, verbo=True):
    dirm = {} #holding name:policy pairs
    ars = {} #holding name:mar pairs
    crs = {} #holding name:cr pairs
    print scen
    scen.initSim()
    nexp = noshexp(scen, config.tadj)
    prob = scen.makeProblem()

    if "OBSA/CR" in custom or "EMSR/CR" in custom:
       prob.prepare()
       u = prob.findu()
       gamma = prob.gamma(u)
       x  = prob.policyCR(u,gamma)
       vc = sum(x[1:])
       if verbo: print "OBSA/CR: VC=", vc, "ratio:", gamma
       if verbo: print "x=", x#, "fCR=", prob.findCR(x)
       #prob.dump()
       #print input("test a policy:")
    if "OBSA/CR" in custom:
       crs["OBSA/CR"] = nexp.gamma(x, verbo, False)
       ars["OBSA/CR"] = nexp.regret(x, verbo, False)
       np = NestedPolicy(x, prob)
       scen.modifyParam(np)
       np = PolicyBed(np, config.BINO) #CR/OBSA
       dirm["OBSA/CR"] = np

    #below was used to debug "gamma > 1.0"
    #ext = scen.extendScen()
    #pex = ext.makeProblem()
    #pex.prepare()
    #u = pex.findu()
    #gamma = pex.gamma(u)
    #x = pex.policyCR(u, gamma)
    #print "ratix:", gamma, "AUTH SEATS:", sum(x[1:])
    #print "x=", x, "fCR=", pex.findCR(x)

    if "EMSR/CR" in custom:
       x = scen.policyEMSR(vc)
       crs["EMSR/CR"] = nexp.gamma(x, verbo, False)
       ars["EMSR/CR"] = nexp.regret(x, verbo, False)
       if verbo: print "EMSR/CR: Allocated Seats", sum(x[1:])
       if verbo: print "x=", x, "CR:=", prob.findCR(x)
       xp = NestedPolicy(x, prob)
       scen.modifyParam(xp)
       xp = PolicyBed(xp, config.BINO) #EMSR/CR
       dirm["EMSR/CR"] = xp

    if "OBSA/AR" in custom or "EMSR/AR" in custom:
       prob.prepare()
       u = prob.findut()
       regret = prob.regret(u)
       x  = prob.policyAR(u)
       vc = sum(x[1:])
       if verbo: print "OBSA/AR:", vc, "regret:", regret 
       if verbo: print "x=", x

    if "OBSA/AR" in custom:
       crs["OBSA/AR"] = nexp.gamma(x, verbo, False)
       ars["OBSA/AR"] = nexp.regret(x, verbo, False)
       ap = NestedPolicy(x, prob)
       scen.modifyParam(ap)
       ap = PolicyBed(ap, config.BINO) #OBSA/AR
       dirm["OBSA/AR"] = ap

    if "EMSR/AR" in custom:
       crs["EMSR/AR"] = nexp.gamma(x, verbo, False)
       ars["EMSR/AR"] = nexp.regret(x, verbo, False)
       x = scen.policyEMSR(vc)
       if verbo: print "EMSR/AR:", sum(x[1:])
       if verbo: print "x=", x, "CR:=", prob.findCR(x)
       xp = NestedPolicy(x, prob)
       scen.modifyParam(xp)
       xp = PolicyBed(xp, config.BINO) #EMSR/CR
       dirm["EMSR/CR"] = xp

    x = scen.policyEMSR(scen.C)
    crs["EMSR/NO"] = nexp.gamma(x, verbo, False)
    ars["EMSR/NO"] = nexp.regret(x, verbo, False)
    if verbo: print "EMSR/NO:", sum(x[1:])
    if verbo: print "x=", x 
    xo = NestedPolicy(x, prob)
    scen.modifyParam(xo)
    xo = PolicyBed(xo, config.BINO) #EMSR/CR
    dirm["EMSR/NO"] = xo

    x = scen.policyStaticDP()
    if verbo: print "DP/LBH:", sum(x[1:])
    if verbo: print "x=", x
    crs["DP/LBH"] = nexp.gamma(x, verbo, False)
    ars["DP/LBH"] = nexp.regret(x, verbo, False)
    tp = NestedPolicy(x, prob)
    scen.modifyParam(tp)
    tp = PolicyBed(tp, config.BINO) #DP/LBH
    dirm["DP/LBH"] = tp

    pm = pmean(scen.p)

    #print "Offline Optimal Policy!"
    mp = OffOptPolicy(prob)
    scen.modifyParam(mp)
    mp = PolicyBed(mp, config.BINO) #Off/Opt
    dirm["Off/Opt"] = mp

    #vc = scen.newsVCL(1.-pm, 1)
    if config.BINO: vc = scen.newsVCL() #test VCL
    else: vc = scen.newsVCU() #test VCU

    x = scen.policyEMSR(vc)
    crs["EMSR/NV"] = nexp.gamma(x, verbo, False)
    ars["EMSR/NV"] = nexp.regret(x, verbo, False)
    if verbo: print "EMSR/NV:", sum(x[1:])
    if verbo: print "x=", x
    ep = NestedPolicy(x, prob)
    scen.modifyParam(ep)
    ep = PolicyBed(ep, config.BINO)#EMSR/NV
    dirm["EMSR/NV"] = ep

    prop = scen.makeProblem()
    prop.p = (0.0, 0.0) #the MS paper model
    prop.C = vc #Note: the vc from EMSR/NV
    prop.prepare()
    u = prop.findu()
    gamma = prop.gamma(u)
    x  = prop.policyCR(u,gamma)
    if verbo: print "CRSA/NV:", sum(x[1:]), "ratio:", gamma
    if verbo: print "x=", x
    crs["CRSA/NV"] = nexp.gamma(x, verbo, False)
    ars["CRSA/NV"] = nexp.regret(x, verbo, False)
    pp = NestedPolicy(x, prob)
    scen.modifyParam(pp)
    pp = PolicyBed(pp, config.BINO)#CRSA/NV
    dirm["CRSA/NV"] = pp
    
    vc = scen.virtucapSL1(config.svl)
    x = scen.policyEMSR(vc)
    if verbo: print "EMSR/SL1:  Allocated Seats", vc
    if verbo: print "x=", x
    crs["EMSR/SL"] = nexp.gamma(x, verbo, False)
    ars["EMSR/SL"] = nexp.regret(x, verbo, False)
    crs["EMSR/S1"] = crs['EMSR/SL']
    ars["EMSR/S1"] = ars['EMSR/SL']
    s1p = NestedPolicy(x, prob)
    scen.modifyParam(s1p)
    s1p = PolicyBed(s1p, config.BINO)#EMSR/SL
    dirm["EMSR/SL"] = s1p #default to SL1
    dirm["EMSR/S1"] = s1p

    vc = scen.virtucapSL2(config.svl)
    x = scen.policyEMSR(vc)
    crs["EMSR/S2"] = nexp.gamma(x, verbo, False)
    ars["EMSR/S2"] = nexp.regret(x, verbo, False)
    if verbo: print "EMSR/SL2:  Allocated Seats", vc
    if verbo: print "x=", x
    sp = NestedPolicy(x, prob)
    scen.modifyParam(sp)
    sp = PolicyBed(sp, config.BINO)#EMSR/SL
    dirm["EMSR/S2"] = sp

    ff = FCFSPolicy(vc, prob)
    scen.modifyParam(ff)
    ff = PolicyBed(ff, config.BINO)
    dirm["FCFS"] = ff 

    if "HCR/OSA" in custom or "EMSR/HCR" in custom:
        avg = [(prob.L[i]+prob.U[i])/2. for i in range(prob.m+1)]
        std = [prob.U[i]-prob.L[i] + avg[i]/3. 
						for i in range(prob.m+1)]
        nexp.initMRAS(avg, std, *config.cfgCE())
        nexp.search(config.iters, config.HCR)
        while nexp.research(config.iters, config.HCR): pass
        else: print "Search Done:", nexp.best
        x = nexp.best[:]
        vc = sum(x[1:])

	import math
    if "HCR/OSA" in custom:
        crs["HCR/OSA"] = nexp.gamma(x, verbo, False)
        ars["HCR/OSA"] = nexp.regret(x, verbo, False)
        assert math.fabs(crs["HCR/OSA"]-nexp.gamma(x, verbo))<1e-9
        zp = NestedPolicy(x, prob)
        scen.modifyParam(zp)
        dirm["HCR/OSA"] = PolicyBed(zp, config.BINO)
        if verbo: print "HCR/OSA: x=", x

    if "EMSR/HCR" in custom:
        x = scen.policyEMSR(vc)
        crs["EMSR/HCR"] = nexp.gamma(x, verbo, False)
        ars["EMSR/HCR"] = nexp.regret(x, verbo, False)
        if verbo: print "EMSR/HCR:", sum(x[1:]), x
        #nexp.gamma(x, verbo)
        zp = NestedPolicy(x, prob)
        scen.modifyParam(zp)
        dirm["EMSR/HCR"] = PolicyBed(zp, config.BINO)

    if "HAR/OSA" in custom or "EMSR/HAR" in custom:
        avg = [(prob.L[i]+prob.U[i])/2. for i in range(prob.m+1)]
        std = [prob.U[i]-prob.L[i] + avg[i]/3. 
						for i in range(prob.m+1)]
        nexp.initMRAS(avg, std, *config.cfgCE())
        nexp.search(config.iters, config.HAR)
        while nexp.research(config.iters, config.HAR): pass
        else: print "Search Done:", nexp.best
        x = nexp.best[:]
        vc = sum(x[1:])

    if "HAR/OSA" in custom:
        crs["HAR/OSA"] = nexp.gamma(x, verbo, False)
        ars["HAR/OSA"] = nexp.regret(x, verbo, False)
        assert math.fabs(ars["HAR/OSA"]-nexp.regret(x, verbo))<1e-9
        zp = NestedPolicy(x, prob)
        scen.modifyParam(zp)
        dirm["HAR/OSA"] = PolicyBed(zp, config.BINO)
        if verbo: print "HAR/OSA: x=", x

    if "EMSR/HAR" in custom:
        x = scen.policyEMSR(vc)
        crs["EMSR/HAR"] = nexp.gamma(x, verbo, False)
        ars["EMSR/HAR"] = nexp.regret(x, verbo, False)
        if verbo: print "EMSR/HAR:", sum(x[1:]), x
        zp = NestedPolicy(x, prob)
        scen.modifyParam(zp)
        dirm["EMSR/HAR"] = PolicyBed(zp, config.BINO)

    return dirm, ars, crs

def scenaQuant(scen, custom, DISP=None, bRATIO=False, reps=50, qnts=100):
    if bRATIO: #this case don't show Off/Opt
        if 'Off/Opt' in custom: 
             del custom[custom.index('Off/Opt')]
        custom.append('Off/Opt')
    offid = len(custom)-1

    dirm,ars,crs = getPolicies(scen, custom)
    src = scen.makeSource()
    print "Source p:", src.p

    pls = [dirm[pon] for pon in custom]
    qus = [[[0 for i in range(qnts)] 
            for j in range(reps)] 
            for k in range(len(pls))]

    for j in range(reps): 
      for i in range(qnts):
        seq = src.nextSeq(config.LBH, COR)
        nop = src.nextRate(ptype) 
        if config.BINO: binoSeq(seq, nop)
        for k in range(len(pls)):
            qus[k][j][i] = pls[k].execute(seq, nop)[0]
        if bRATIO:
            for k in range(offid):
                qus[k][j][i] /= qus[offid][j][i]
    rav = range(0, 100, 9)
    qav = [[0.0 for t in rav] for k in range(len(pls))]
    for k in range(len(pls)):
        if bRATIO and k == offid: continue
        for i in range(reps):
            qus[k][i].sort()
            for s, t in enumerate(rav):
                qav[k][s] += qus[k][i][t]
        for s in range(len(qav[k])):
            qav[k][s] /= float(reps)

    params = {'axes.labelsize': 10,
             'text.fontsize': 10,
             'legend.fontsize': 10,
             'xtick.labelsize': 8,
             'ytick.labelsize': 8,
			 'lines.markeredgewidth':0.2,
			 'lines.markersize':2}
    rcParams.update(params)

    fig = figure(figsize=(6,4))
    vvv = [ t+1 for t in rav ]
    for k in range(len(pls)):
        if bRATIO and k == offid: continue
        plot(vvv, qav[k], style[k], label=custom[k])
    legend(loc=0, numpoints = 2)#, markerscale = 0.9)
    xlabel('Percentile')
    if bAbsolute: ylabel('Average net revenues')
    else: ylabel('Relative Performance to Off/Opt (%)')

    if DISP!=None: savefig(DISP+'-qnt.eps')
    else: show()

def avgRevSvcCI(scen, custom, reps=40, qnts=5000):
    rev = [[0.0 for j in range(reps)] for k in range(len(custom))]
    svc = [[0.0 for j in range(reps)] for k in range(len(custom))]
    for j in range(reps):
      resu,lvl,los,his,ars,crs,stds = \
			scenaSim(scen, qnts, custom, False)
      for k in range(len(custom)):
        rev[k][j] = resu[k][0]
        svc[k][j] = resu[k][2]

    rav = [0.0 for k in range(len(custom))]
    rtd = [0.0 for k in range(len(custom))]
    sav = [0.0 for k in range(len(custom))]
    std = [0.0 for k in range(len(custom))]
    Z = 1.98 # 95% z-score
    ff='%.2f' #print format
    for k in range(len(custom)):
      for j in range(reps):
        rav[k] += rev[k][j]
        rtd[k] += rev[k][j]*rev[k][j]
        sav[k] += svc[k][j]
        std[k] += svc[k][j]*svc[k][j]

      rav[k] /= float(reps)
      rtd[k] = sqrt(rtd[k]/float(reps) - rav[k]*rav[k])
      sav[k] /= float(reps)
      std[k] = sqrt(std[k]/float(reps) - sav[k]*sav[k])
      print custom[k],'&', ff%rav[k], '&', ff%rtd[k], 
      print '& [', ff%(rav[k]-rtd[k]*Z),',',ff%(rav[k]+rtd[k]*Z),'] &',
      print ff%sav[k], '&', ff%std[k], 
      print '& [', ff%(sav[k]-std[k]*Z),',',ff%(sav[k]+std[k]*Z),']\\\\'

def revSvcCI(scen, custom, reps=5000, silent=False, ras=None):
    """revenue & service Confidence Intervals"""
    if ras is None:
       dirm,ars,crs = getPolicies(scen, custom, False)
       src = scen.makeSource()
       #print "Source p:", src.p
       pls = [dirm[pon] for pon in custom]
       ras = [[0.0 for j in range(reps)] for k in range(len(pls))]
   
       for j in range(reps): 
           seq = src.nextSeq(config.LBH, COR)
           nop = src.nextRate(ptype) 
           if config.BINO: binoSeq(seq, nop)
           for k in range(len(pls)):
               ras[k][j] = pls[k].execute(seq, nop)

    rav = [0.0 for k in range(len(custom))]
    rtd = [0.0 for k in range(len(custom))]
    sav = [0.0 for k in range(len(custom))]
    std = [0.0 for k in range(len(custom))]

    Z = 1.98 # 95% z-score
    ff='%.2f' #print format
    for k in range(len(custom)):
      for j in range(reps):
        rav[k] += ras[k][j][0]
        rtd[k] += ras[k][j][0]*ras[k][j][0]
        if ras[k][j][1] <= 0.0: continue
        denials = ras[k][j][1]*10000./scen.C
        sav[k] += denials
        std[k] += denials*denials

      rav[k] /= float(reps)
      rtd[k] = sqrt(rtd[k]/float(reps) - rav[k]*rav[k])
      rtd[k] /= sqrt(reps)
      sav[k] /= float(reps)
      std[k] = sqrt(std[k]/float(reps) - sav[k]*sav[k])
      std[k] /= sqrt(reps)
      if silent: continue
      print custom[k],'&', ff%rav[k], '&', ff%rtd[k], 
      print '& [', ff%(rav[k]-rtd[k]*Z),',',ff%(rav[k]+rtd[k]*Z),'] &',
      print ff%sav[k], '&', ff%std[k], 
      print '& [', ff%(sav[k]-std[k]*Z),',',ff%(sav[k]+std[k]*Z),'] \\\\'

    if silent: return rav, rtd, sav, std

def revSvcCI_rev1format(scens, custom, reps=5000, fscen=None):
    """output format according to revision 1. this is done for rev2."""
    ravs, rtds, savs, stds = [],[],[],[]
    if fscen: fscen = open(fscen,'rb')
    nscen = 0
    for scen in scens:
       ras = None
       if fscen: ras = pickle.load(fscen)
       if not scen: continue
       rav, rtd, sav, std = \
          revSvcCI(scen, custom, reps, True, ras)
       ravs.append(rav)
       rtds.append(rtd)
       savs.append(sav)
       stds.append(std)
       nscen += 1

    for k in range(len(custom)):
       print custom[k],
       for i in range(nscen):
         rev1format(k, ravs[i], rtds[i], savs[i], stds[i])
       print '\\\\' 

def rev1format(k, rav, rtd, sav, std):
    Z = 1.98  # 95% z-score
    def sigdif(a,da,b,db):
       return abs(a-b) > (da+db)*Z

    def revmark(k):
       """compare with rav[0],rav[1]: 
       if diff from both: $^{\ddag}$ --3
       if only from 0: $^{\dag}$ --2
       if only from 1: $^{\S}$ --1
       other wise: '' -- 0 """
       marks = ('', '$^{\S}$', '$^{\dag}$', '$^{\ddag}$')
       idx = 0
       if sigdif(rav[k],rtd[k],rav[0],rtd[0]):
          idx += 2
       if sigdif(rav[k],rtd[k],rav[1],rtd[1]):
          idx += 1
       return marks[idx]

    def svcmark(k):
       """compare with sav[0],sav[1]: 
       if diff from both: $^{\checkmark}$
       otherwise: '' """
       d0 = sigdif(sav[k],std[k],sav[0],std[0])
       d1 = sigdif(sav[k],std[k],sav[1],std[1])
       return '$^{\checkmark}$' if d0 and d1 else ''

    print ('& [%.2f, %.2f]%s')%(rav[k]-rtd[k]*Z, rav[k]+rtd[k]*Z, revmark(k)),
    print ('& [%.2f, %.2f]%s')%(sav[k]-std[k]*Z, sav[k]+std[k]*Z, svcmark(k))

#Test if the differences are significantly different from zero
def diffRevTest(scen, custom, reps=50, qnts=100):
    dirm,ars,crs = getPolicies(scen, custom)
    src = scen.makeSource()
    print "Source p:", src.p

    pls = [dirm[pon] for pon in custom]
    qus = [[0.0 for j in range(reps)] for k in range(len(pls))]

    for j in range(reps): 
      for i in range(qnts):
        seq = src.nextSeq(config.LBH, COR)
        nop = src.nextRate(ptype) 
        if config.BINO: binoSeq(seq, nop)
        for k in range(len(pls)):
            qus[k][j] += pls[k].execute(seq, nop)[0]

    rav = [0.0 for k in range(len(pls)-1)]
    std = [0.0 for k in range(len(pls)-1)]
    print "Average revenues from", qnts, "runs are assumed ~ Normal distr."
    print "One sample t-test on the differences btw avg. revenues, DF =", reps-1
    for k in range(len(pls)-1):
      for j in range(reps):
        difa = (qus[k+1][j]-qus[k][j])/float(qnts)
        rav[k] += difa
        std[k] += difa*difa
      rav[k] /= float(reps)
      std[k] = sqrt(std[k]/float(reps) - rav[k]*rav[k])
      print custom[k+1], "-", custom[k], "avg:", rav[k], "std:", std[k], "t:", rav[k]/std[k]*sqrt(reps)

def scenaSim(scen, reps, custom, verbo=True, fscen=None):
    if verbo: print ">>>> New Scenario: ID =", scen.nid
    dirm,ars,crs = getPolicies(scen, custom, verbo)
    src = scen.makeSource()
    if verbo: print "Source p:", src.p

    pls = [dirm[pon] for pon in custom]
    ars = [ars[pon] for pon in custom]
    crs = [crs[pon] for pon in custom]
    key = [None for pon in pls] #key performance
    lvl = [None for p in pls]
    los = [None for pon in pls]
    his = [None for pon in pls]
    vrev = [None for pon in pls]
    vstd = [None for pon in pls] 
    vlos = [None for pon in pls]
    vhis = [None for pon in pls]
    vemp = [None for pon in pls]
    vbum = [None for pon in pls]

    seq = src.nextSeq(config.LBH, COR)
    nop = src.nextRate(ptype) 
    if config.BINO: binoSeq(seq, nop)
    for k in range(len(pls)):
        obsv = pls[k].execute(seq, nop)
        los[k] = his[k] = obsv[0]

    ras = [[0.0 for i in range(reps)] 
                for k in range(len(pls))]
    for i in range(reps):
        seq = src.nextSeq(config.LBH, COR)
        nop = src.nextRate(ptype) 
        if config.BINO: binoSeq(seq, nop)

        for k in range(len(pls)):
            ras[k][i] =obsv= pls[k].execute(seq, nop)
            if obsv[0] < los[k]: los[k] = obsv[0]
            if obsv[0] > his[k]: his[k] = obsv[0]

    def mustd(data, std4mu=False): #mu and std
        no = float(len(data))
        mu = sum(data)/no
        sg = sum(d*d for d in data)/no
        std = sqrt((sg-mu*mu)/(no if std4mu else 1.0))
        return mu, std

    def mustd_minus(data, std4mu=False): #mu and std
        no = float(len(data))
        mu = sum(data)/no
        sg = sum(p*p for p in (mu-d if d<mu else 0 for d in data))/no
        return mu, sqrt(sg/(no if std4mu else 1.0))

    def bootstrap(revs, statis, itsN):
        stt = [0 for i in range(itsN)]
        for i in range(itsN): #resample with replacement
            resam = [revs[randint(0,reps-1)] for d in revs]
            resam.sort()
            stt[i] = statis(resam)
        if type(stt[0]) in (int, float): return mustd(stt)
        return tuple(mustd(data) for data in zip(*stt))

    if config.percentile: #find percentiles
        ipercent = reps*config.percentile/100
        def jackknife_percentile(revs, percentile):
            idx, rho = reps*percentile/100+1, percentile/100.
            delta = revs[idx+1] - revs[idx]
            return delta*sqrt((1-rho)*rho*(reps-1))

        def lohi_percentiles(revs):
            return revs[ipercent], revs[reps-ipercent]

        def lohistd_boot(revs):
            return revs[ipercent], revs[reps-ipercent],\
							mustd(revs, True)[1]

        remp = 100.0/scen.C 
        for k in range(len(pls)):
            revenues = sorted(r[0] for r in ras[k])
            vlos[k], vhis[k], vstd[k] = bootstrap(
				revenues, lohistd_boot, 500)
            los[k], his[k] = lohi_percentiles(revenues)
            vrev[k] = mustd(revenues, True)
			#empty if <0, bumps if >0.
            vbum[k] = mustd([( d if d>0 else 0) 
					for r,d in ras[k]], True)
            vemp[k] = mustd([(-d if d<0 else 0) 
					for r,d in ras[k]], True)
            rbum = 10000./(scen.C - vemp[k][0] + vbum[k][0])
            vbum[k] = [x*rbum for x in vbum[k]]
            vemp[k] = [x*remp for x in vemp[k]]

    #find variance for each method 
	print "Table: statistics for revenues"
	print "=============================="
	print "Meth.|E[R]|Std.|5\%|std|95%|std" 
    for k in range(len(pls)):
        print custom[k], '&%.0f &%.0f'%(vrev[k][0], 2*vrev[k][1]),
        if config.percentile: #print percentiles
            print "&%.0f"%los[k], "&%.0f"%(2*vlos[k][1]), 
            print "&%.0f"%his[k], "&%.0f"%(2*vhis[k][1]),
        print '\\\\'
    #compare k, k+t, use Common Random Number
    for k in range(len(pls)):
        print custom[k],
        for j in range(len(pls)):
            mu, std = mustd([rk[0]-rj[0] 
               for rk,rj in zip(ras[k],ras[j])], True)
            print '&%.2f'%(mu/std), 
        print '\\\\'
    #save the scenario details
    if fscen: pickle.dump(ras, fscen)

    import math
    bench = pls[0].res[0]/reps/100.0 #for percentage
    for k in range(len(pls)):
        key[k] = pls[k].average(reps)
        lvl[k] = pls[k].levels()
        #addjustments
        if not bAbsolute: key[k][0]/= bench
        #scen.C - avg(empty seats) + avg(denials)=avg(eff bookings)
        key[k][2] *= 10000.0/(scen.C-key[k][1]+key[k][2])
        key[k][1] *= remp
        if verbo: print custom[k], key[k]
        if verbo: print "+==>>", vemp[k], vbum[k]
        if verbo: print custom[k], lvl[k]
        #assert math.fabs(key[k][1]-vemp[k][0])<1e-9
        #assert math.fabs(key[k][2]-vbum[k][0])<1e-9
	stds = {'vrev':vrev, 'vstd':vstd, 'vlos':vlos, 'vhis':vhis,
					'vemp':vemp, 'vbum':vbum}
    return key,lvl,los,his,ars,crs,stds

def barChart(data, label):
    locats = [i for i in range(1, len(data)+1)]
    bar(locats, data)
    xticks([x + 0.4 for x in locats], label) 

#barChart([2, 3, 1], ['Tom', 'Barbara', 'Mark'])

##legend(*args, **kwargs): Function 
##The location codes are
##  'best' : 0,
##  'upper right'  : 1, (default)
##  'upper left'   : 2,
##  'lower left'   : 3,
##  'lower right'  : 4,
##  'right'        : 5,
##  'center left'  : 6,
##  'center right' : 7,
##  'lower center' : 8,
##  'upper center' : 9,
##  'center'       : 10,
##If none of these are suitable, loc can be a 2-tuple giving x,y
##in axes coords, ie,
##  loc = 0, 1 is left top
##  loc = 0.5, 0.5 is center, center
##and so on. The following kwargs are supported:

##isaxes=True           # whether this is an axes legend
##numpoints = 4         # the number of points in the legend line
##prop = FontProperties(size='smaller')  # the font property
##pad = 0.2             # the fractional whitespace inside the legend border
##markerscale = 0.6     # the relative size of legend markers vs. original
##shadow                # if True, draw a shadow behind legend
##labelsep = 0.005      # the vertical space between the legend entries
##handlelen = 0.05      # the length of the legend lines
##handletextsep = 0.02  # the space between the legend line and legend text
##axespad = 0.02        # the border between the axes and legend edge

def glidePrint(sa, sb, fa, fb, grids):
    glide = ScenaGlide(sa,sb)
    nm = len(meth)
    vvv = [k/grids for k in range(fa, fb)]
    for k in range(len(vvv)):
        sinf = glide.getGlide(vvv[k])
        print k, "Demand factor:", sinf.demandFactor()
        vvv[k] = sinf.nid
        scenaPrint(sinf)

def glideSim(sa, sb, fa, fb, grids, reps,
             DISP, custom=meth):
    print "config.BINO: ", config.BINO
    print "config.tadj: ", config.tadj
    print "config.LBH: ",  config.LBH
    print "meth: ", custom

    glide = ScenaGlide(sa,sb)
    vvv = [k/grids for k in range(fa, fb)]
    gld = [glide.getGlide(v) for v in vvv]
    enuSim(gld, reps, DISP, custom)

def policyFile(DISP): 
    return  "policies-"+DISP

def modiSim(gld, reps, custom):
    '''modify the scen and simulate'''
    resu, lvl,los,his ,ars,crs, stds= \
		scenaSim(sinf, reps, custom)

def enuSim(gld, reps, DISP, custom=meth):
    nm = len(custom)
    relat = [[] for i in range(nm)]
    waste = [[] for i in range(nm)]
    bumps = [[] for i in range(nm)]
    miner = [[] for i in range(nm)]
    maxer = [[] for i in range(nm)]
    polis = [[] for i in range(nm)]
    crsmt = [[] for i in range(nm)]
    arsmt = [[] for i in range(nm)]
    vrev =  [[] for i in range(nm)]
    vstd =  [[] for i in range(nm)]
    vlos =  [[] for i in range(nm)]
    vhis =  [[] for i in range(nm)]
    vemp =  [[] for i in range(nm)]
    vbum =  [[] for i in range(nm)]
    vvv = [sinf.nid for sinf in gld]

    scenout = None
    if DISP.endswith('fscen.pkl'):
       scenout = open(DISP,'wb')
       DISP=DISP[0:-9]+'.pkl'

    for k, sinf in enumerate(gld):
        print k, "Demand factor:", sinf.demandFactor()
        resu,lvl,los,his,ars,crs, stds = \
			scenaSim(sinf, reps, custom, fscen = scenout)
        for i in range(nm):
            relat[i].append(resu[i][0])
            waste[i].append(resu[i][1])
            bumps[i].append(resu[i][2])
            miner[i].append(los[i])
            maxer[i].append(his[i])
            arsmt[i].append(ars[i])
            crsmt[i].append(crs[i])
            polis[i].append(lvl[i])
            vrev[i].append(stds['vrev'][i])
            vstd[i].append(stds['vstd'][i])
            vlos[i].append(stds['vlos'][i])
            vhis[i].append(stds['vhis'][i])
            vemp[i].append(stds['vemp'][i])
            vbum[i].append(stds['vbum'][i])
    stds = {'vrev':vrev,'vstd':vstd,'vlos':vlos,'vhis':vhis,
					'vemp':vemp, 'vbum':vbum}

    if scenout: 
        scenout.close()
        print "Scenario details saved:", scenout

    #save data for later use, see 'loadResults()' below
    output = open(DISP, 'wb')
    pickle.dump(custom, output)
    pickle.dump(vvv, output)
    pickle.dump(relat, output)
    pickle.dump(waste, output)
    pickle.dump(bumps, output)
    pickle.dump(miner, output)
    pickle.dump(maxer, output)
    pickle.dump(arsmt, output)
    pickle.dump(crsmt, output)
    pickle.dump(stds, output)
    output.close()
    print "Result saved to file:", DISP
    polfile = policyFile(DISP)
    output = open(polfile, 'wb')
    pickle.dump(custom, output)
    pickle.dump(vvv, output)
    pickle.dump(polis, output)
    output.close()
    print "Policies saved to file:", polfile

from pprint import pprint
def drawPolicies(DISP, xlab, custom, vvv, polis):
    nmeth = len(custom)
    nlevel = len(polis[0][0])
    transp, aggreg = [], []
    for i in range(nmeth):
        #transpose polis[i]
        transp.append( zip( *polis[i] ) )
        aggreg.append( [[sum(limits[j:]) 
				for j in range(nlevel)] 
				for limits in polis[i]] )
        aggreg[i] = zip( *aggreg[i] )

    fig = figure(figsize=(6,4))
    for i in range(nmeth):
        plot(vvv, aggreg[i][0], style[i], label=custom[i],
				markerfacecolor='None')
        for j in range(1, nlevel):
            plot(vvv, aggreg[i][j], style[i],
				markerfacecolor='None')
    adjust_style(fig)
    legend(numpoints = 1, loc=0)
    xlabel(xlab)
    if config.buckets:
        ylabel("buckets for fare classes")
    else: 
        ylabel("nested buckets for fare classes")

    if DISP!=None: 
        savefig(DISP+'-bucks.eps')
        print "File saved: %s-bucks.eps"%DISP

    for j in range(nlevel):
        fig = figure(figsize=(6,4))
        for i in range(nmeth):
            plot(vvv, transp[i][j], style[i], label=custom[i],
				markerfacecolor='None')
        adjust_style(fig)
        legend(numpoints = 1, loc=0)
        xlabel(xlab) 
        if config.buckets:
           ylabel("bucket for fare class-%i"%(j+1))
        else: 
           ylabel("nested bucket for fare class-%i"%(j+1))
        if DISP!=None: 
           savefig(DISP+'-bucks-%i.eps'%(j+1))
           print "File saved: %s-bucks-%i.eps"%(DISP,j+1)

from matplotlib.patches import Ellipse, Rectangle
def drawCIcircles(fig, vvv, data, style, Z=1.96):
   '''draw confidence intervals as circles on plot'''
   w,h = fig.get_size_inches()
   ax = fig.gca()
   aspect = h/(w*ax.get_data_ratio())*3 #six sigma
   #print w, h, aspect, data
   ells=[Ellipse(xy=(x, y), 
             width=2*std*Z*aspect, height=2*std*Z)
       for x, (y, std) in zip(vvv, data)]
   for ell in ells:
      ell.set_facecolor('none')
      ell.set_alpha(0.5)
      ell.set_edgecolor(style[0])
      ell.set_linewidth(0.4)
      #ell.set_linestyle('dashed')
      ax.add_artist(ell)

def drawCIbars(fig, vvv, data, style, Z=1.96):
   '''draw confidence intervals as error bars on plot'''
   errorbar(vvv, [y for y, std in data], capsize=4, 
		yerr=[std*Z for y, std in data], fmt=None, 
		ecolor=style[0], elinewidth=0.5, barsabove=True)

drawCI = drawCIcircles
drawCI = drawCIbars

def adjust_style(fig, subadj=True):
    ax = gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position(('axes',-0.02))
    #ax.spines['bottom'].set_position(('axes',-0.02))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    for o in fig.findobj(): o.set_clip_on(False)
    if subadj: subplots_adjust(left=0.15, right=0.95, 
					bottom=0.12, top=0.92)

def drawFigs(DISP, xlab, custom, vvv, relat, waste, bumps,
		miner, maxer, arsmt, crsmt, conf={}, legloc=(0,0,0)):
    rcParams.update({'axes.labelsize': 10,
             'text.fontsize': 10,
             'legend.fontsize': 10,
             'xtick.labelsize': 8,
             #'xtick.direction': 'out',
             'ytick.labelsize': 8,
             'ytick.direction': 'out',
             'axis.linewidth': .5,
			 'lines.linewidth': 0.3,
			 'lines.markeredgewidth':0.2,
			 #'lines.markerfacecolor':'None',
			 'lines.markersize':4})
    nm = len(custom)
    fig = figure(figsize=(6,4))
    autoscale(tight=True) 
    for i in range(nm):
        plot(vvv, relat[i], style[i], 
				markerfacecolor='None', label=custom[i])

	# draw confidence intervals as circles here
    cir = conf.get("vrev", None)
    for i in range(nm):
        if cir: drawCI(fig, vvv, cir[i], style[i])
    adjust_style(fig)

    #for i in range(nm):
    #    plot(vvv, miner[i], style[i])
    #    plot(vvv, maxer[i], style[i])
    legend(loc=legloc[0], numpoints = 1)#, markerscale = 0.8)
    xlabel(xlab)
    if bAbsolute: ylabel('Average net revenues')
    else: ylabel('Relative Performance to EMSR/CR (%)')
    #title('Relative Performance to EMSR/CR')

    if DISP!=None: 
        savefig(DISP+'1.eps')
        print "File saved: %s1.eps"%DISP
    
    fig = figure(figsize=(6,4))
    autoscale(tight=True) 
    for i in range(nm):
        plot(vvv, waste[i], style[i], #label=custom[i])
				markerfacecolor='None', label=custom[i])
    cir = conf.get("vemp", None)
    for i in range(nm):
        if cir: drawCI(fig, vvv, cir[i], style[i])
    adjust_style(fig)
    legend(loc=legloc[1], numpoints = 1)#, markerscale = 0.9)
    xlabel(xlab)
    ylabel('Average unused inventory (per 100)')
    #title('Average Empty Seats per 100')
    if DISP!=None: 
        savefig(DISP+'2.eps')
        print "File saved: %s2.eps"%DISP
    
    fig = figure(figsize=(6,4))
    autoscale(tight=True) 
    for i in range(nm):
        plot(vvv, bumps[i], style[i], #label=custom[i])
				markerfacecolor='None', label=custom[i])
    cir = conf.get("vbum", None)
    for i in range(nm):
        if cir: drawCI(fig, vvv, cir[i], style[i])
    adjust_style(fig)
    legend(loc=legloc[2], numpoints = 1)#, markerscale = 0.9)
    xlabel(xlab)
    ylabel('Average service denials (per 10,000)')
    #title('Average Bumpings per 10,000')
    if DISP!=None: 
        savefig(DISP+'3.eps')
        print "File saved: %s3.eps"%DISP

    fig = figure(figsize=(6,4))
    autoscale(tight=True) 
    for i in range(nm):
        plot(vvv, miner[i], style[i], #label=custom[i])
				markerfacecolor='None', label=custom[i])
	# draw confidence intervals as circles here
    cir = conf.get("vlos", None)
    for i in range(nm):
        if cir: drawCI(fig, vvv, cir[i], style[i])
    adjust_style(fig)
    legend(loc=legloc[2], numpoints = 1)#, markerscale = 0.9)
    xlabel(xlab)
    if config.percentile: ylabel(
			'Observed %ith percentile of revenue'%config.percentile)
    else: ylabel('Minimal Observed Revenue')
    #title('Average Bumpings per 10,000')
    if DISP!=None: 
        savefig(DISP+'4.eps')
        print "File saved: %s4.eps"%DISP

    fig = figure(figsize=(6,4))
    autoscale(tight=True) 
    for i in range(nm):
        plot(vvv, maxer[i], style[i], #label=custom[i])
				markerfacecolor='None', label=custom[i])
	# draw confidence intervals as circles here
    cir = conf.get("vhis", None)
    for i in range(nm):
        if cir: drawCI(fig, vvv, cir[i], style[i])
    adjust_style(fig)
    legend(loc=legloc[2], numpoints = 1)#, markerscale = 0.9)
    xlabel(xlab)
    if config.percentile: ylabel(
			'Observed %ith percentile of revenue'%
			(100-config.percentile))
    else: ylabel('Maximal Observed Revenue')
    #title('Average Bumpings per 10,000')
    if DISP!=None: 
        savefig(DISP+'5.eps')
        print "File saved: %s5.eps"%DISP

    fig = figure(figsize=(6,4))
    autoscale(tight=True) 
    for i in range(nm):
        plot(vvv, arsmt[i], style[i], #label=custom[i])
				markerfacecolor='None', label=custom[i])
    adjust_style(fig)
    legend(loc=legloc[2], numpoints = 1)#, markerscale = 0.9)
    xlabel(xlab)
    ylabel('Maximal Absolute Regret')
    #title('Average Bumpings per 10,000')
    if DISP!=None: 
        savefig(DISP+'6.eps')
        print "File saved: %s6.eps"%DISP

    fig = figure(figsize=(6,4))
    autoscale(tight=True) 
    for i in range(nm):
        plot(vvv, crsmt[i], style[i], #label=custom[i])
				markerfacecolor='None', label=custom[i])
    adjust_style(fig)
    legend(loc=legloc[2], numpoints = 1)#, markerscale = 0.9)
    xlabel(xlab)
    ylabel('Competitive Ratio')
    #title('Average Bumpings per 10,000')
    if DISP!=None: 
        savefig(DISP+'7.eps')
        print "File saved: %s7.eps"%DISP

    cir = conf.get("vstd", None)
    if not cir: return
    fig = figure(figsize=(6,4))
    autoscale(tight=True) 
    for i in range(nm):
        stds = [x for x,y in cir[i]]
        plot(vvv, stds, style[i], #label=custom[i])
				markerfacecolor='None', label=custom[i])
    legend(loc=legloc[2], numpoints = 1)#, markerscale = 0.9)
	# draw confidence intervals as circles here
    for i in range(nm):
        drawCI(fig, vvv, cir[i], style[i])
    adjust_style(fig)

    xlabel(xlab)
    ylabel('Standard deviation of mean revenue')
    #title('Average Bumpings per 10,000')
    if DISP!=None: 
        savefig(DISP+'8.eps')
        print "File saved: %s8.eps"%DISP

def drawSubFigs(DISP, xlab, custom, vvv, relat, waste, bumps,
		miner, maxer, arsmt, crsmt, conf={}, legloc=(0,0,0)):
    rcParams.update({'axes.labelsize': 6,
             'text.fontsize': 5,
             'legend.fontsize': 5,
             'xtick.labelsize': 4,
             #'xtick.direction': 'out',
             'ytick.labelsize': 5,
             'ytick.direction': 'out',
             'axis.linewidth': .5,
			 'lines.linewidth': 0.3,
			 'lines.markeredgewidth':0.2,
			 #'lines.markerfacecolor':'None',
			 'lines.markersize':4})
    nm = len(custom)
    figure(figsize=(6,4))
    fig = subplot(221)
    for i in range(nm):
        plot(vvv, relat[i], style[i], 
				markerfacecolor='None', label=custom[i])
	# draw confidence intervals as circles here
    cir = conf.get("vrev", None)
    for i in range(nm):
        if cir: drawCI(fig, vvv, cir[i], style[i])
    adjust_style(fig, False)

    legend(loc=legloc[0], numpoints = 1)#, markerscale = 0.8)
    xlabel(xlab)
    if bAbsolute: ylabel('Average net revenues')
    else: ylabel('Relative Performance to EMSR/CR (%)')
    #title('Relative Performance to EMSR/CR')

    fig = subplot(223)
    cir = conf.get("vstd", None)
    for i in range(nm):
        stds = [x for x,y in cir[i]]
        plot(vvv, stds, style[i], #label=custom[i])
				markerfacecolor='None', label=custom[i])
    legend(loc=legloc[2], numpoints = 1)#, markerscale = 0.9)
	# draw confidence intervals as circles here
    for i in range(nm):
        drawCI(fig, vvv, cir[i], style[i])
    adjust_style(fig, False)
    xlabel(xlab)
    ylabel('Standard deviation of mean revenue')

    fig = subplot(224)
    for i in range(nm):
        plot(vvv, miner[i], style[i], #label=custom[i])
				markerfacecolor='None', label=custom[i])
	# draw confidence intervals as circles here
    cir = conf.get("vlos", None)
    for i in range(nm):
        if cir: drawCI(fig, vvv, cir[i], style[i])
    adjust_style(fig, False)
    legend(loc=legloc[2], numpoints = 1)#, markerscale = 0.9)
    xlabel(xlab)
    if config.percentile: ylabel(
			'Observed %ith percentile of revenue'%config.percentile)
    else: ylabel('Minimal Observed Revenue')

    fig = subplot(222)
    for i in range(nm):
        plot(vvv, maxer[i], style[i], #label=custom[i])
				markerfacecolor='None', label=custom[i])
	# draw confidence intervals as circles here
    cir = conf.get("vhis", None)
    for i in range(nm):
        if cir: drawCI(fig, vvv, cir[i], style[i])
    adjust_style(fig, False)
    legend(loc=legloc[2], numpoints = 1)#, markerscale = 0.9)
    xlabel(xlab)
    if config.percentile: ylabel(
			'Observed %ith percentile of revenue'%
			(100-config.percentile))
    else: ylabel('Maximal Observed Revenue')
    tight_layout()
    if DISP!=None: 
        savefig(DISP+'0.eps')
        print "File saved: %s0.eps"%DISP
 
def loadResults(filen):
    pkl_file = open(filen, 'rb')
    custom = pickle.load(pkl_file)
    vvv = pickle.load(pkl_file)
    relat = pickle.load(pkl_file)
    waste = pickle.load(pkl_file)
    bumps = pickle.load(pkl_file)
    miner = pickle.load(pkl_file)
    maxer = pickle.load(pkl_file)
    arsmt = pickle.load(pkl_file)
    crsmt = pickle.load(pkl_file)
    stds = pickle.load(pkl_file)
    pkl_file.close()
    return custom,vvv,relat,waste,bumps,miner,maxer,arsmt,crsmt,stds

def loadPolicies(filen):
    pkl_file = open(policyFile(filen), 'rb')
    custom = pickle.load(pkl_file)
    vvv = pickle.load(pkl_file)
    polis = pickle.load(pkl_file)
    pkl_file.close()
    return custom, vvv, polis

###########################################################
#Below are new codes for the 3rd part of my dissertation. #
###########################################################

### The experiment assumes Binomial noshow distribution.
# With Binomial distribution, the offline optimal can be
#  found by a greedy algorithm as described below:
#    For any given input sequence: keep choosing the 
#    next highest fare request, and see if by adding
#    this request would incease expected net revenue.
#  ############################## 
# For each input stream in the reduced input stream set,
# precompute its offline revenue. Also, precompute the
# expected penalty for each total bookings, which can be
# done recursively for Binomial noshows: 
#       g(y+1) = g(y) + p*V*P{Z(y)>=C}.
# The result above can be generalized to Uniform-Binomial
# noshow distributions, see ../newexp.pdf for details.
##########################################################

#take the following parameters:
#std: standard deviation of denial cost
#dep: probability of service denial event (Service Level I)
#der: (expected service denials) / (expected show-ups) 
def penadj(tp, smul, svl, svlp):
    def noadjus(std, dep, der): 
        return 0.0

    def stdplus(std, dep, der): 
        return std*smul

    def depplus(std, dep, der): 
        return max(0,dep-svl)*svlp

    def derplus(std, dep, der): 
        return max(0,der-svl)*svlp

    #print "adjust type:", tp
    if(tp<1): return noadjus
    if(tp<2): return stdplus
    if(tp<3): return depplus
    if(tp<4): return derplus
    return None

class penub: #penulty function with Uniform-Binomial dist.
    def __init__(me, noshowp, V, C, B, penadj):
        C = int(round(C))
        B = int(round(B))
        me.q = (1.-noshowp[1],1.-noshowp[0]) #show prob.
        if noshowp[1] - noshowp[0] < 1e-8: 
            me.s = newvc.BinomialMass(sum(me.q)/2, C, B)
        else:
            me.s = newvc.UnifBinoMass(me.q[0],me.q[1], C, B) 
        me.V = float(V)
        me.C = C
        me.B = B
        me.f = [0. for i in range(C, B)]
        me.K = C #currently available function value
        me.padj = penadj

    def g(me, t):
        if(t >= me.B): raise "penulf: out of range!"
        if(t <= me.C): return 0.0
        if(t <= me.K): return me.f[t-me.C]
        for i in range(me.K+1, t+1):
           gav, suv, dep = 0.0,0.0,0.0
           me.s.expand()
           for j in range(me.C+1, i+1):
              pmf = me.s.choose(j)
              dep += pmf
              rem = (j-me.C)*pmf
              gav += rem
              suv += (j-me.C)*rem
           suv = sqrt(suv - gav*gav) #std(v)
           der = t*sum(me.q)/2.0
           der = me.padj(suv, dep, gav/der)
           me.f[i-me.C] = me.V*(gav+der)
           #print sum, me.V*sum/(i+1)
        me.K = t
        return me.f[t-me.C]

    def gdep(me, t): #depricated
        if(t >= me.B): raise "penulf: out of range!"
        if(t <= me.C): return me.f[0]
        if(t <= me.K): return me.f[t-me.C]
        for i in range(me.K, t):
           sum = 0.0
           me.s.expand()
           for j in range(me.C+1, i+2):
              sum += j*me.s.choose(j)
           me.f[i-me.C+1] = me.f[i-me.C] + me.V*sum/(i+1)
           #print sum, me.V*sum/(i+1)
        me.K = t
        return me.f[t-me.C]

    def simug(me, t, reps=10000): #estimate g(t) by simulation
        tq, tv, ts = 0.0, 0.0, 0.0
        for i in range(reps):
            q = me.q[0]+random()*(me.q[1]-me.q[0])
            tq = 0
            for j in range(t):
                if random() <= q: tq += 1
            tq = max(0, tq - me.C) * me.V
            #print (i,tq),
            tv += tq
            ts += tq * tq
        tv /= reps
        return tv, sqrt(ts/reps - tv*tv)

    def h(me,t):
        t0, t1 = int(floor(t)), int(ceil(t))
        if t0 == t1: return me.g(int(t))
        return (t-t0)*me.g(t1) + (t1-t)*me.g(t0)


#if noshow rate is uniform
class penun:
    def __init__(me, noshowp, V, C, penadj):
        me.q = (1.-noshowp[1],1.-noshowp[0]) #show probability
        me.d = noshowp[1] - noshowp[0] 
        me.V = float(V)
        me.C = C
        me.padj = penadj

    def g(me, t):
        if t*me.q[1] <= me.C: return 0.0
        q0 = max(me.q[0], me.C/float(t))
        ra, st = 1.0, 0.0
        if me.d>0: 
            ra = (me.q[1]-q0)/me.d
            st = (me.q[1]*t-me.C)**3-(q0*t-me.C)**3
            st /= (3.*t*me.d) #E(v^2)
        vb = (t*(q0+me.q[1])/2.0-me.C)*ra
        st -= vb*vb
        if st > 0.0: st = sqrt(st)
        else: st = 0.0
        der = t*sum(me.q)/2.0
        return me.V*(vb+me.padj(st,ra,vb/der))

    def simug(me, t, reps=10000): #estimate g(t) by simulation
        tq, tv, ts = 0.0, 0.0, 0.0
        for i in range(reps):
            q = me.q[0]+random()*(me.q[1]-me.q[0])
            tq = max(0, t*q - me.C) * me.V
            #print (i,tq),
            tv += tq
            ts += tq * tq
        tv /= reps
        return tv, sqrt(ts/reps - tv*tv)

    def h(me,t):
        t0, t1 = floor(t), ceil(t)
        if t0 == t1: return me.g(t)
        return (t-t0)*me.g(t1) + (t1-t)*me.g(t0)

#test code for penub and penun
def testPenultyFunc():
    f = penun((0.2,0.2), 1, 100, penadj(2,config.smul, config.svl, config.svlp))
    g = [f.g(i) for i in range(100, 200)]
    plot(g)
    u,s = f.simug(140)
    print f.g(140), u+s*config.smul

    f = penub((0.2,0.4), 1, 100, 200, penadj(2,config.smul, config.svl, config.svlp))
    g = [f.g(i) for i in range(100, 200)]
    plot(g)
    u,s = f.simug(150)
    print f.g(150), u+s*config.smul
    show()

#testPenultyFunc()

class noshexp:
    def __init__(me, info, tadj):
        assert isinstance(info, SimScena)
        me.scen = [] 
        me.extr = []

        us = info.uset if hasattr(info,'uset') else config.uset 
        mean = [(x+y)/2. for x,y in zip(info.U, info.L)]
        me.U = (0,)+tuple(int(m+(x-m)*us+.9) 
						for m,x in zip(mean, info.U))
        me.L = (0,)+tuple(max(0, int(m+(x-m)*us))
						for m,x in zip(mean, info.L))

        if config.BINO:
            B = 1+int(sum(me.U))
            if B < info.C:
               B = info.C
               print "Warning: sum(U) < C!"
            me.ff = penub(info.p, info.V, info.C, B, 
                penadj(tadj,config.smul, config.svl, config.svlp))
        else:
            me.ff = penun(info.p, info.V, info.C,
                penadj(tadj,config.smul, config.svl, config.svlp))
        h = sum(info.p)/2.0
        h = 1.0 - h + info.beta*h
        me.f = (info.V*(1-info.p[0]),) + info.f #fares
        me.f = [h*fi for fi in me.f]
        me.m = info.m
        me.C = info.C
        me.baseK = config.basemult*sum(f*(u+l)/2. 
				for f,u,l in zip(info.f, info.U, info.L))

        #[[k, Dk, R*], ]
        for k in range(info.m,0,-1):
            R, T = me.Roff(k, int(me.L[k]))
            me.extr.append((k, int(me.L[k]), R, T))
            for d in range(int(me.L[k]), int(me.U[k])):
                R, T = me.Roff(k, d)
                me.scen.append( (k, d, R, T ) )
        d = int(me.U[1])
        R, T = me.Roff(1, d)
        me.extr.append((1, d, R, T))
        me.scen.append((1, d, R, T))

    def Roff(me, k, d):
        T = [0 for i in range(me.m+1)]
        i,Di,R = 0,0,0.0
        while (i <= me.m):
            while Di < 1:
                i = i+1
                if i>me.m: break
                Di = int(me.L[i])
                if i==k: Di = d
                if i>k: Di = int(me.U[i])
            if Di < 1: break
            r = R + me.ff.g(T[0]) + me.f[i] - me.ff.g(T[0]+1)
            if R >= r: break
            R, Di = r, Di-1
            T[0] += 1
            T[i] += 1
        return R, T

    def Ron(me, k, d, x):
        T,Di,bi,R = 0, 0, 0, 0.0
        for i in range(me.m, 0, -1):
            if i<k: Di = int(me.L[i])
            if i==k: Di = d
            if i>k: Di = int(me.U[i])
            if x[i] > 0: bi += x[i]
            ax = min(bi, T+Di)
            R += (ax-T) * me.f[i]
            T = ax
        return R - me.ff.h(T)

    def gamma(me, x, verbose=False, easy=True):
        gg = 1.00000000001
        bs = None
        scen = me.extr if easy else me.scen
        for s in scen:
            if s[2] <= 0: continue
            gs = (me.baseK+me.Ron(s[0],s[1],x))/(me.baseK+s[2])
            if gs < gg: gg, bs = gs, s
        if verbose: print "Gamma:", gg, "Bind Scen:", bs
        return gg

    def regret(me, x, verbose=False, easy=True):
        "use the max absolute regret -- MAR"
        gg = -0.0000000001
        bs = None
        scen = me.extr if easy else me.scen
        for s in scen:
            gs = s[2] - me.Ron(s[0],s[1],x)
            if gs > gg: gg, bs = gs, s
        if verbose: print "Regret:", gg, "BindScen:", bs, "Ron:", bs[2]-gg
        return gg

    def wregret(me, x, verbose=False): #weighted regret
        "experiment with a from of weighted absolute regret"
        gg, sg = 0.0, 0.0
        for s in me.scen:
            gs = s[2] - me.Ron(s[0],s[1],x)
            gg += gs * gs
            sg += gs
        if verbose: print "Regret:", gg, "BindScen:", bs, "Ron:", bs[2]-gg
        return gg/sg

    #the codes below are for global optimization by MRAS/CEM.
    #a policy x is generated, with x[0] denoting its CR.
    #each x[i] is generated by a normal distribution.
    def initMRAS(me, u, s, z, e, alpha): #init mu, sigma, sample size
        me.u = u #u,s must be list
        me.s = s
        me.e = e #elite size, z: sample size
        me.alpha = float(alpha) #smooth
        me.beta = 1. - alpha
        me.z = [[0 for i in range(0,me.m+1)] for j in range(0,z)]
        me.gauss = Random().gauss
        me.best = [0.0 for i in range(0,me.m+1)]

    def sample(me, x, fobj, cut): #generate i^th sample
        for j in range(1, cut):
            x[j] = me.gauss(me.u[j], me.s[j])
            if x[j] < 0: x[j] = 0 
            elif x[j] > me.U[j]+.5: x[j] = me.U[j]+.5
        for j in range(cut, me.m+1): x[j] = 0 
        x[0] = fobj(x)

    #The following are equivalent: li = [(1,2),(3,1)]
    #li.sort(key=lambda x:x[1], reverse=True )
    #li.sort(lambda x, y: cmp(x[1],y[1]), reverse=True) 
    #li.sort(lambda x, y: cmp(y[1],x[1])) 
    def update(me, wet):
        avg=[0.0 for i in range(me.m+1)]
        std=[0.0 for i in range(me.m+1)]
        for j in range(me.e):
            for i in range(me.m+1):
                z = me.z[j][i]*wet[j]
                avg[i] += z
                std[i] += z*me.z[j][i]
        #avg[1] = min(avg[1], me.U[1])
        for i in range(me.m+1): 
            if avg[i] < 0.0: avg[i] = 0
            std[i] -= avg[i]*avg[i]
            if std[i] < 1e-9: std[i] = 1e-9 
            std[i] = sqrt(std[i])
        for i in range(me.m+1): 
            me.u[i] = me.u[i]*me.alpha + avg[i]*me.beta
            me.s[i] = me.s[i]*me.alpha + std[i]*me.beta

    def update(me, _): #MX way
        avg=[0.0 for i in range(me.m+1)]
        std=[0.0 for i in range(me.m+1)]
        for j in range(me.e):
            for i in range(me.m+1):
                z = me.z[j][i]
                avg[i] += z
                std[i] += z*z
        for i in range(me.m+1): 
            if avg[i] < 0.0: avg[i] = 0
            avg[i] /= float(me.e)
            std[i] /= float(me.e)
            std[i] -= avg[i]*avg[i]
            z = avg[i] - me.u[i]
            std[i] += z*z
            if std[i] < 1e-9: std[i] = 1e-9 
            std[i] = sqrt(std[i])
        for i in range(me.m+1): 
            me.u[i] = me.u[i]*me.alpha + avg[i]*me.beta
            me.s[i] = me.s[i]*me.alpha + std[i]*me.beta

    def research(me, maxit, tobj):
        best = me.best[:]
        #if sum(math.fabs(a-b) for a, b 
		#	in zip(me.best[1:], me.u[1:]))<0.5: #precision ctrl
        #    return
        me.u = me.best
        me.s = [i+1 for i in me.u] #or i/2.
        me.best = [0.0 for i in range(0,me.m+1)]
        me.search(maxit, tobj)
        import math
        bigdif = math.fabs(best[0]-me.best[0]) > max(
						0.001, 0.005*best[0])
			#or sum(math.fabs(a-b) for a,b in 
			#		zip(best[1:], me.best[1:])) > 0.1*me.m
        if (tobj==noshowConfig.HCR and me.best[0]<best[0]) or \
               (tobj>noshowConfig.HCR and me.best[0]>best[0]): 
                 me.best = best
        return bigdif

    def cutsearch(me, maxit, tobj, cut=None):
        if cut is None: cut = me.m + 1
        if tobj == noshowConfig.HCR:
            fobj = me.gamma 
        elif tobj == noshowConfig.HAR: 
            fobj = me.regret
        elif tobj == 2: 
            fobj = me.wregret
        else: raise "BAD argument 'tobj': " + str(tobj)
        me.best[0]=fobj(me.best) if tobj!=2 else fobj(me.best,True)
        fmt = '%.3f' if tobj <= 0 else '%.1f'
        #print "Searching with obj: ", fobj

        for it in range(0, maxit):
            for x in me.z: me.sample(x, fobj, cut)
            #sort samples and calculate weights
            me.z.sort(key=lambda x: x[0], reverse=(tobj==0))
            wet=[1.+max(0,me.z[j][0]) for j in range(me.e)]
            if tobj==1: wet=[1.0/wet[j] for j in range(me.e)]
            swt = sum(wet)
            for j in range(me.e): wet[j] /= swt
            #update the distribution
            me.update(wet)
            #record the best
            #print me.z[0], me.best
            #fobj(me.z[0], verbose=True)
            if (tobj==noshowConfig.HCR and me.z[0][0]>me.best[0])\
               or (tobj>noshowConfig.HCR and me.z[0][0]<me.best[0]): 
                   me.best = me.z[0][:]
            if it % config.vsearch == 0 : 
               print it, '&', fmt%me.best[0],
               for bik in range(1, me.m+1):
                  print '&','%.1f'%me.best[bik],
               print '\\\\'
        print "Best Found:", me.best
        print "MRAS Mean: ", me.u
        print "MRAS Std:  ", me.s

    def search(me, maxit, tobj, cut=None):
        best,u,s = None, me.u, me.s #u,s must be list
        for mincut in range(2, me.m+2):
            if sum(me.U[1:mincut])>me.C: break
        for cut in range(mincut, me.m+2):
            me.u, me.s = u[:], s[:] 
            me.best = [0.0 for i in range(0,me.m+1)]
            print "### Searching with cut =", cut
            print "### Bounds:", me.L, me.U
            print "### Std:", me.s
            me.cutsearch(maxit, tobj, cut)
            if best is None or \
			   (tobj==noshowConfig.HCR and me.best[0]>best[0]) or\
               (tobj>noshowConfig.HCR and me.best[0]<best[0]): 
                 best = me.best[:]
        me.best = best
        print "Best Found (Multicut):", me.best
#NOTE: use reduce to get a sum of f(x) for all x in a list:
#For example, let f(x) = x*x, and li is a list of numbers.
#The python code goes like this:
#####|| reduce(lambda s,x: s+x*x, li, 0)
#Here the function reduce() was passed a third parameter as
#the default/starting value.
#For more info: http://docs.python.org/tut/node7.html
