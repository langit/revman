from pylab import plot, show, legend, xlabel, ylabel, title

class pico: #for 2 fares only
    def __init__(me, n, b, V, R, f, p, kk=1):
        me.vbs=False
        me.n = n #capacity
        me.b = b #beta: refund penalty
        me.V = V #denial penalty
        me.R = R #Range
        me.f = f #fare
        assert(f[0] >= f[1])
        if kk<0: #thoro
            kk = -kk
            me.sp = [(i,j) for i in range(R[0][0],R[0][1]+1)
                     for j in range(R[1][0],R[1][1]+1)]
        else:
            me.sp = [(i,j) for i in R[0] for j in R[1]]
        me.p = p #noshow rates
        #no-show rates combinations: (p0 or p1, pi), (pi, po or p1)
        dd = float(kk)
        def pcom(i,j): return p[0]+(p[1] - p[0])*(i/dd), p[0]+(p[1] - p[0])*(j/dd)
        me.nr = [pcom(i,j) for i in range(kk+1) for j in range(kk+1)
                 if  i in (0,kk) or j in (0,kk)]
        me.oo = [me.offrev(i,j) for i in me.sp for j in me.nr]
        U1 = R[0][1]+1
        U2 = R[1][1]+1
        me.xx = [(i,j) for i in range(0,U1) for j in range(0,U2)]

    #consistent fare ordering: don't nest in wrong way with it?
    def fconsist(me, vbs=False): 
        # f_i*(1-p+beta*p)/(1-p) = f_i*(1-beta+beta/(1-p))
        #OR simply: f_i*(1-(1-beta)*p)?
        fsmall = [f*(1-me.b+me.b/(1-me.p[0])) for f in me.f]
        flarge = [f*(1-me.b+me.b/(1-me.p[1])) for f in me.f]
        if(vbs): print(fsmall); print(flarge)
        smallf = [f*(1-(1-me.b)*me.p[1]) for f in me.f]
        largef = [f*(1-(1-me.b)*me.p[0]) for f in me.f]
        if(vbs): print(smallf); print(largef)
        return fsmall[0] - flarge[1] # > 0 is OK

    def forder(me, nr, ftype=3):
        """test fare order: if high is >= low, return 1.
             nr: no-show rates
             ftype=0: expected revenue per request
             ftype=1: expected revenue per seat (off opt)
             ftype=2: expected revenue per request with denial.
             ftype=3: all 0, 1, 2
        """
        ef = [fr[0]*(1-(1-me.b)*fr[1]) for fr in zip(me.f,nr)]
        cv0 = 1
        if ef[0]<ef[1]: cv0 = 0
        if ftype==0: return cv0
        cv1 = 1
        if ef[0]*(1.-nr[1])<ef[1]*(1.-nr[0]): cv1 = 0
        if ftype==1: return cv1
        cv2 = 1
        if ef[0]-me.V*(1-nr[0])<ef[1]-me.V*(1-nr[1]): cv2 = 0
        if ftype==2: return cv2
        if ftype==3: return cv0*cv1*cv2
        return cv1+cv2+cv3

    def bookrev(me, w, nr): #revenue from booking records
        rev = 0.0 #revenue so far
        use = 0.0 #seats used
        if me.vbs: print("revenue from booking prof.", w, "with noshow rates", nr)
        for k,r,f in zip(w,nr,me.f):
            rev += k*f*(1-r+me.b*r)
            use += k*(1-r)
        pen = max(0,use - me.n)*me.V
        if me.vbs: print("rev: ", rev, "  use: ", use, "pen: ",pen)
        return rev - pen

    def offrev(me, s, r): #offline optimal
        w = [0.,0.]
        left = me.n
        for i in range(0,2):
            w[i] = min(left/(1-r[i]), s[i])
            left -= w[i]*(1-r[i])
        rvn = me.bookrev(w, r)
        w = [0.,0.]
        left = me.n
        for i in range(1,-1, -1):
            w[i] = min(left/(1-r[i]), s[i])
            left -= w[i]*(1-r[i])
        rev = me.bookrev(w, r)
        return max(rvn, rev)

    def seprev(me, buck, s, r):
        w = [min(x,t) for x,t in zip(buck, s)]
        return me.bookrev(w,r)

    def gsep(me, buck, gr=False): #get the CR of a sep. bucket
        ron = [me.seprev(buck, s, r) for s in me.sp for r in me.nr]
        gon = [r/t for r,t in zip(ron, me.oo)]
        assert(max(gon) <= 1.0)
        if gr: return gon
        else: return min(gon)

    def sepg(me):
        gon = [me.gsep(x) for x in me.xx]
        ii = gon.index(max(gon))
        return gon[ii], me.xx[ii]

    def neslbh(me, nest, s, r, rvs): #LBH
        if rvs: #then it is actually HBL
           nest = (nest[1],nest[0]) 
           s = (s[1],s[0])
        lo = min(nest[1], s[1])
        le = nest[1] - lo
        hi = min(nest[0]+le, s[0])
        if rvs: hi, lo = lo, hi
        if me.vbs: print("LBH: ", (hi, lo) )
        return me.bookrev((hi,lo), r)

    def neshbl(me, nest, s, r, rvs): #HBL
        if rvs: #then it is actually LBH
           nest = (nest[1],nest[0]) 
           s = (s[1],s[0])
        hi = min(s[0], nest[0]+nest[1])
        he = max(hi - nest[0], 0)
        lo = min(s[1], nest[1] - he)
        if rvs: hi, lo = lo, hi
        if me.vbs: print("HBL: ", (hi, lo) )
        return me.bookrev((hi,lo), r)

    def nesrev(me, nest, s, r, rvs): 
        lbh = me.neslbh(nest,s,r, rvs)
        hbl = me.neshbl(nest,s,r, rvs)
        return min(lbh, hbl) #the smaller of LBH and HBL 

    def gnes(me, nest, rvs, gr=False): #get the CR of a nest bucket
        ron = [me.nesrev(nest,s,r,rvs) for s in me.sp for r in me.nr]
        gon = [r/t for r,t in zip(ron, me.oo)]
        assert(max(gon) <= 1.0)
        if gr: return gon
        else:  return min(gon)

    def nesg(me, rvs):
        gon = [me.gnes(x, rvs) for x in me.xx]
        ii = gon.index(max(gon))
        return gon[ii], me.xx[ii]

#myp = pico(30, 0.8, 20, ((4,21), (10,20)), (5.,4.), (0.0,0.2))
#print("LBH: ", myp.neslbh((2,3),(3,3),(.05,.15)))
#print("HBL: ", myp.neshbl((2,3),(3,3),(.05,.15)))
#print(myp.nr, myp.sp)
#print("Offline Opt: ", myp.oo)
#print("Opt. sep. g: ", myp.sepg())
#print("Opt. nes. g: ", myp.nesg())

def criticbeta(f, p):
    return 1-(f[0]-f[1])/(f[0]*p[1]-f[1]*p[0])

import random as rr

def randscen(n):
    n = rr.randint(4,n)
    ll = rr.randint(1,n//2)
    lh = rr.randint(1+n//2, n+1)
    hl = rr.randint(1, n//2)
    hh = rr.randint(1+n//2, n+1)
    V = 2+5*rr.random()
    f2 = rr.random()
    p0 = 0.2*rr.random()
    p1 = p0 + 0.1 + 0.2*rr.random()
    beta = rr.random()
    return n, ll, lh, hl, hh, V, f2, p0, p1, beta

def randmyp(mit, cbeta=False):
    for i in range(mit):
        n, ll, lh, hl, hh, V, f2, p0, p1, beta =randscen(40)
        if cbeta: beta = criticbeta((1,f2),(p0,p1))
        if beta < 0: continue
        mp = pico(n, beta, V, ((hl, hh), (ll, lh)), (1.,f2), (p0, p1))
        if mp.fconsist()<0.0: #this is needed to ensure correctness of
            print "!",          #  the offline optimal. Is that right?
        sepg = mp.sepg()[0]
        nesg = mp.nesg(False)[0]
        nesv = mp.nesg(True)[0]
        nesg = max(nesg, nesv)
        if sepg - nesg > 1e-12:
            print("Found!", sepg-nesg)
            print((n, ll, lh, hl, hh, beta, V, f2, p0, p1))
            break
        else: print(sepg-nesg, " = " , sepg, " - ", nesg)

def randmip(mit, K, cbeta=False):
    for i in range(mit):
        n, ll, lh, hl, hh, V, f2, p0, p1, beta =randscen(40)
        if cbeta: beta = criticbeta((1,f2),(p0,p1))
        if beta < 0: continue
        mp = pico(n, beta, V, ((hl, hh), (ll, lh)), (1.,f2), (p0, p1), K)
        if mp.fconsist()<0.0: #this is needed to ensure correctness of
            print "!",          #  the offline optimal. Is that right?
        if checkmip(mp)>0: break

def checkmip(mp):
        sepg = mp.sepg()
        gsep=mp.gsep(sepg[1], True)
        ii = gsep.index(sepg[0])
        ip = ii % len(mp.nr)
        pt = mp.nr[ip]
        baog = 0
        if pt[0] not in mp.p or pt[1] not in mp.p:
            print("Buck", ii, mp.nr[ip], gsep[ii], mp.oo[ii])
            baog += 1
        nesg = mp.nesg(False)
        gnes=mp.gnes(nesg[1], False, True)
        jj = gnes.index(nesg[0])
        jp = jj % len(mp.nr)
        pt = mp.nr[jp]
        if pt[0] not in mp.p or pt[1] not in mp.p:
            print("Nest", jj, mp.nr[jp], gnes[jj], mp.oo[jj])
            baog += 1
        nesv = mp.nesg(True)
        vnes=mp.gnes(nesv[1], True, True)
        kk = vnes.index(nesv[0])
        kp = kk % len(mp.nr)
        pt = mp.nr[kp]
        if pt[0] not in mp.p or pt[1] not in mp.p:
            print("Vest", kk, mp.nr[kp], gnes[kk], mp.oo[kk])
            #baog += 1
        if baog > 0:
            print((n, ll, lh, hl, hh, beta, V, f2, p0, p1))
        return baog


ex1 = (37, 9, 15, 10, 31, .2, 4, .8, .05, .3)
ex2 = (13, 1, 7, 3, 10, 0.10, 2.7, 0.95, 0.10, 0.32)
ex3 = (21, 2, 6, 9, 21, 0.51, 5.4, 0.92, 0.12, 0.32)
ex4 = (9, 1, 1, 0, 9, 0.5, 4, 0.9, 0.01, 0.3)
ex5 = (7, 1, 3, 3, 7, 0.2, 6, 0.77, 0.1, 0.37)
ex6 = (4, 1, 1, 1, 4, 0.1, 5.5, 0.877, 0.05, 0.28)
ex7 = (4, 2, 2, 0, 3, 0.3558102194606092, 6.4004154738673558, 0.96343706549834318, 0.15370999086560255, 0.41501753879472358)

ex8 = (11, 1, 6, 4, 10, 0.005, 6., 0.74, 0.1, 0.4)
exx = (11, 1, 6, 4, 10, 0.0047215624306254922, 6.1477013223616623, 0.74145688767277895, 0.13176659694643858, 0.41805916544045257)

ex9 = (14, 6, 11, 0, 8, 0.51687299005508558, 4.7732664378484504, 0.9379846659502189, 0.17920572209926461, 0.43781179813946702)

ex10 = (20, 9, 12, 5, 11, 0.23159509112651033, 5.0899919678163386, 0.99835136953696346, 0.065887199132170213, 0.28970400621898013)

#('Found!', 0.013225130927484341)
ex11= (39, 12, 21, 11, 30, 0.16031035780327574, 6.1631621348012784, 0.98721978776628105, 0.14733451187810648, 0.44685104033225509)
ex12= (39, 12, 21, 11, 30, .15, 6, .8, .05, .35)
ex12b= (39, 12, 21, 11, 30, .15, 6, .8, .1, .4) #Sep. better than pmax(NPL, rNPL)
ex12c= (39, 12, 21, 11, 30, .2, 5, .8, .1, .4) 
ex12d= (39, 12, 21, 11, 30, .375, 6, .8, .1, .4) 

#('Found!', 0.0076925740008755294)
ex13 = (47, 21, 24, 20, 35, 0.3463412485584173, 3.8931346564078408, 0.95574707707549633, 0.11465881085396826, 0.37011052755206114)

#('Found!', 0.00093804487802429115)
ex14=(30, 7, 17, 6, 19, 0.53079162798360258, 4.9155145077897995, 0.91014761295393753, 0.052653552300594098, 0.27108136218417755)

#('Found!', 0.0053265272511607842)
ex15=(26, 6, 20, 10, 14, 0.031988495159048114, 4.3152262374414851, 0.80424349932625339, 0.12926602408868901, 0.42868724828253513)


def randpp(mit, cbeta=True):
  rvs = False
  for i in range(mit):
    n, ll, lh, hl, hh, V, f2, p0, p1, beta=randscen(40)
    mp = pico(n, beta, V, ((hl, hh), (ll, lh)), (1.,f2), (p0, p1))
    if cbeta: beta = criticbeta((1,f2),(p0,p1))
    if beta < 0: continue
    nesg = mp.nesg(rvs)
    gnes=mp.gnes(nesg[1], rvs, True)
    jj = gnes.index(nesg[0])
    if jj%4==1: 
        print(n, ll, lh, hl, hh, beta, V, f2, p0, p1)
        ex = (n, ll, lh, hl, hh, beta, V, f2, p0, p1)
        testSample(ex)
        break

rex1 = (10, 3, 11, 1, 11, 0.91365345176542645, 4.9792487573313418, 0.98512564826449334, 0.050889430344559464, 0.22239590342192439)
#(13, ((11, 11), (0.050889430344559464, 0.22239590342192439)), 0.8067383918265012)
rex2 = (23, 5, 18, 1, 12, 0.34983532898211944, 3.7874855812171386, 0.78805295542249576, 0.16942207325021219, 0.45950335554935084)


def testSample(ex, kk = 1, cbeta=False):
    print("sample", ex)
    n, ll, lh, hl, hh, beta, V, f2, p0, p1 = ex
    if cbeta: beta = criticbeta((1,f2),(p0,p1))
    mp = pico(n, beta, V, ((hl, hh), (ll, lh)), (1.,f2), (p0,p1), kk)
    #print(mp.sp)
    ron = [(s, r) for s in mp.sp for r in mp.nr]
    print len(mp.nr), len(mp.sp), len(ron)
    #print zip(range(len(ron)), ron)
    pes = [mp.nesrev(x,sn[0],sn[1],False)-mp.seprev(x,sn[0],sn[1])
         for x in mp.xx for sn in ron]
    print("NPL ever less than SB if <0:", min(pes))
    #print("!consist? ", mp.fconsist(True))
    sepg = mp.sepg()
    print("Sep. Bucket: ", sepg)
    gsep=mp.gsep(sepg[1], True)
    ii = gsep.index(sepg[0])
    print(ii, ron[ii], gsep[ii], mp.oo[ii]) #worst case

    print("Nest according to fare order")
    rvs = False
    nesg = mp.nesg(rvs)
    print("Nes. Bucket: ", nesg)
    gnes=mp.gnes(nesg[1], rvs, True)
    jj = gnes.index(nesg[0])
    print(jj, ron[jj], gnes[jj], mp.oo[jj]) #worst case
    js, jr = ron[jj]
    mp.vbs=True
    print("LBH: ", mp.neslbh(nesg[1], js, jr, rvs))
    print("HBL: ", mp.neshbl(nesg[1], js, jr, rvs))
    jj -= 1
    print(jj, ron[jj], gnes[jj], mp.oo[jj])
    js, jr = ron[jj]
    print("Online Rev: ", mp.nesrev(nesg[1], js, jr, rvs))
    print("Offline Rev: ", mp.offrev(js, jr))
    mp.vbs=False

    #print("Sep:", jj, gsep[jj])
    #print("Nes:", ii, gnes[ii])

    print("Nest according to reversed fare order")
    rvs = True
    nesv = mp.nesg(rvs)
    print("Nev. Bucket: ", nesv)
    gnev=mp.gnes(nesv[1], rvs, True)
    kk = gnev.index(nesv[0])
    print(kk, ron[kk], gnev[kk], mp.oo[kk]) #worst case
    ks, kr = ron[kk]
    mp.vbs=True
    print("LBH (reversed): ", mp.neslbh(nesv[1], ks, kr, rvs))
    print("HBL (reversed): ", mp.neshbl(nesv[1], ks, kr, rvs))
    mp.vbs=False
    #if kk>1: return
    plot(gsep, 'gs-', label="Separate")
    plot(gnes, 'r+-.',label="NPL")
    plot(gnev, 'b^:', label="reverse NPL")
##    maxoo = max(mp.oo)
##    for i in range(len(mp.oo)): mp.oo[i]/=maxoo
##    plot(mp.oo, label="Off. Opt")
    legend(loc=0, numpoints = 2, markerscale = 0.9)
    title("Comparison of Discrete Optimal Policies")
    xlabel("scenarios: (profiles, noshow rates)")
    ylabel("ratio to the offline optimal")
    show()

if __name__ == "__main__":
    #randmyp(9000, True)
    kk = 11
    #randmip(20, kk)
    exs = (ex1, ex2, ex3, ex4, 
     ex5, ex6, ex7, ex8, ex9, 
      ex11, ex12, ex12b)
    for ex in exs:
        n, ll, lh, hl, hh, beta, V, f2, p0, p1 = ex
        mp = pico(n, beta, V, ((hl, hh), (ll, lh)), (1.,f2), (p0,p1), kk)
        checkmip(mp)
    #for ex in exs: testSample(ex)
    #testSample(ex12b)
    #testSample(ex12b, 21)
    #randpp(200)
    #testSample(ex1, True)
