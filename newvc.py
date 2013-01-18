
class IntRangeMass:
    def __init__(me, lo, hi, pm = None):
        me.lo = lo = int(lo+.5)
        me.hi = hi = int(hi+.5)
        if pm != None:
            me.pm = pm
        else:
            evn = 1.0/(hi - lo + 1)
            me.pm = [evn for s in range(hi - lo + 1)]

    def __mul__(me, u):
        lom = me.lo + u.lo
        him = me.hi + u.hi
        print me.lo, me.hi, u.lo, u.hi, lom, him
        mas = [0.0 for s in range(him - lom +1)]
        for s in range(lom, him+1):
            for t in range(max(me.lo, s-u.hi), min(me.hi, s-u.lo)+1):
##                if t>me.hi or t < me.lo or s-t>u.hi or s-t<u.lo:
##                    print  "our of range: %i, %i"%(t, s-t)
##                pa  = me.pm[t-me.lo]
##                pb = u.pm[s-t-u.lo]
##                pc = mas[s-lom]
                mas[s-lom]+=me.pm[t-me.lo] * u.pm[s-t-u.lo]
        return IntRangeMass(lom, him, mas)

    def toCDF(me):
        if me.isCDF(): return
        for i in range(1, len(me.pm)):
            me.pm[i] += me.pm[i-1]

    def toPM(me):
        if not me.isCDF(): return
        for i in range(len(me.pm)-1, 0, -1):
            me.pm[i] -= me.pm[i-1]

    def isCDF(me):
        return me.pm[len(me.pm)-1] >= 1.0 - 1e-9

    def mass(me, x):
        x = int(x)
        if x < me.lo: return 0.0
        if x > me.hi: return 1.0
        return me.pm[x - me.lo]

##m1 = IntRangeMass(2, 8)
##m2 = IntRangeMass(0, 9)
##m3 = m1 * m2 * m1
##print m3.pm
##from pylab import *
##plot(m3.pm)
##m3.toCDF()
##figure()
##plot(m3.pm)
##show()

def totalCDF(L, U):
    """Produce total demand CDF given
        the low and upper bounds."""

    mm = IntRangeMass(L[0], U[0])
    for i in range(1, len(L)):
        mm *= IntRangeMass(L[i], U[i])
    mm.toCDF()
    return mm

class CombSide: 
    """This gives the Yang Hui Triangle."""

    def __init__(me, out, maxout=100):
        me.out = 0
        me.com = [1 for i in range(maxout+1)]
        for i in range(out): me.expand()

    def expand(me):
        for i in range(me.out, 0, -1):
            me.com[i] += me.com[i-1]
        me.out += 1

    def shrink(me):
        for i in range(1, me.out):
            me.com[i] -= me.com[i-1]
        me.out -= 1

    def choose(me, i):
        """return (me.out choose i)"""
        if i < 0 or i > me.out:
            return 0
        return me.com[i]

    def coms(me):
        return me.com[:me.out+1]

##com = CombSide(160, 200)
##print com.coms()
##com.expand()
##print com.out, 'choose 5 is: ', com.choose(5)
##com.shrink()
##print com.coms()

class BinomialMass:
    """( N choose k ) p^k * (1-p)^(N-k)"""
    def __init__(me, p, out, maxout=100, CDF=False):
        me.out = 0
        me.x = p
        me.com = [1 for i in range(maxout+1)]
        if CDF: me.CDF = 1.
        else: me.CDF = 0.
        for i in range(out): me.expand()

    def expand(me):
        y = 1. - me.x
        me.com[me.out+1] = me.com[me.out] * me.x + me.CDF*y
        for i in range(me.out, 0, -1):
            me.com[i] = me.com[i-1]*me.x + me.com[i]*y
        me.com[0] = me.com[0] * y
        me.out += 1

    def normalize(me, fact):
        for i in range(me.out+1):
            me.com[i] /= fact

    def shrink(me, t=1):
        """The inverse method of me.expand().
            To avoid numerical UNSTABLENESS,
            we need to make sure that me.x <= 0.5!
            so we need to call me.reverse() first,
            then me.reverse() back after shrink."""
        UNSTABLE = ( me.x > 0.5 )
        if UNSTABLE: me.reverse()
        y = 1.0/(1. - me.x)
        delta = me.x * y
        for k in range(t):
            for i in range(1, me.out):
                me.com[i] -= me.com[i-1] * delta
                me.com[i-1] *= y
            me.out -= 1
            me.com[me.out] *= y
            if me.CDF > 0: me.com[me.out] = 1.0
        if UNSTABLE: me.reverse()

    def reverse(me):
        """by reversing, we have me.x = 1.0 - me.x!"""
        me.x = 1.0 - me.x
        for i in range((me.out+1)/2):
            j = me.out - i
            elem = me.com[i]
            if me.CDF < 1.0:
                me.com[i] = me.com[j]
                me.com[j] = elem
            else:
                me.com[i] = 1. - me.com[j-1]
                me.com[j-1] = 1. - elem
 
    def choose(me, i):
        """return (N choose i) p^i(1-p)^(N-i)"""
        if i < 0: return 0.
        if i > me.out: return me.CDF
        return me.com[i]

    def coms(me):
        return me.com[:me.out+1]

##N = 480
##p = 0.2
##k = int(p*N)
##com = BinomialMass(p, N, N+1)
##print sum(com.coms())
##com.expand()
##print com.out, 'choose', k, 'is: ', com.choose(k)
##com.shrink()
##cop = BinomialMass(p, N, N+1)
##import math
##print "summation:", sum(com.coms())
##print sum([math.fabs(com.com[i] - cop.com[i]) for i in range(com.out+1)])

def testBinoMassCDF():
    com=BinomialMass(0.6, 20, 30)
    cum=BinomialMass(0.6, 20, 30, True)
    com.shrink(3)
    cum.shrink(3)
    for i in range(com.out+1):
        print "sum:", sum(com.com[0:i+1]), "CDF: ", cum.choose(i)
    cum.reverse()
    for i in range(com.out+1):
        print "sum:", sum(com.com[com.out-i:com.out+1]), "CDF: ", cum.choose(i)

###Check that the sum of two Ind. Binomial R.V.
###of the same p but different N is another
###Binomial R.V. with same p but sum of N

##p = 0.8
##bin1 = BinomialMass(p, 20, 20)
##m1 = IntRangeMass(0, bin1.out, bin1.coms())
##bin2 = BinomialMass(p, 30, 30)
##m2 = IntRangeMass(0, bin2.out, bin2.coms())
##m3 = m1 * m2
##bin3 = BinomialMass(p, 50, 50)
##import math
##print sum([math.fabs(bin3.com[i] - m3.pm[i]) for i in range(bin3.out+1)])

### >>> 2.21576530819e-016
##
##from pylab import *
##mm=totalCDF((15, 64), (31, 128))
##mm.toPM()
##plot(mm.pm)
##show()


class UnifBinoMass:
    """First draw p uniformly, then Binomial on p."""

    def __init__(me, p0, p1, out, maxout=100):
        me.bin0 = BinomialMass(p0, out+1, maxout+1, True)
        me.bin1 = BinomialMass(p1, out+1, maxout+1, True)
        me.delta = p1 - p0
        me.out = out 

    def expand(me):
        me.bin0.expand()
        me.bin1.expand()
        me.out += 1

    def normalize(me, fact):
        me.bin0.normalize(fact)
        me.bin1.normalize(fact)

    def shrink(me):
        me.bin0.shrink()
        me.bin1.shrink()
        me.out -= 1

    def choose(me, i):
        sum = me.bin0.choose(i)-me.bin1.choose(i)
        sum /= (me.out+1)*me.delta
        return sum

def getquant(pmf, lo, hi, quant):
    rq = [lo-1,hi]
    pr = 0.0
    for i in range(lo, hi+1):
        pr += pmf.choose(i)
        if pr >= quant and rq[0]<lo:
            rq[0] = i
        if pr <= 1. - quant:
            rq[1] = i+1
        #print rq, pr
    return rq

def test_getquant():
    ub = UnifBinoMass(0.7, 0.9, 100)
    rr = getquant(ub, 0, 100, 0.05)
    pm = [ub.choose(i) for i in range(0,101)]
    print rr, sum(pm[:rr[0]]), sum(pm[rr[1]+1:]), sum(pm)
    un = BinomialMass(0.8, 100)
    rr = getquant(un, 0, 100, 0.05)
    pm = [un.choose(i) for i in range(0,101)]
    print rr, sum(pm[:rr[0]]), sum(pm[rr[1]+1:]), sum(pm)

test_getquant()

#ub = UnifBinoMass(0.7, 0.9, 50)
#pm = [ub.choose(i) for i in range(0,51)]
#print pm[50] 
#from pylab import *
#plot(pm)

#bn = BinomialMass(0.8, 50)
#pm = [bn.choose(i) for i in range(0,51)]
#plot(pm)

#show()

